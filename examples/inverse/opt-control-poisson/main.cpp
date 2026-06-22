#include <petsctao.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/assembly/BoundaryDofLayout.hpp>
#include <femx/assembly/BoundaryResidualEquation.hpp>
#include <femx/assembly/EnzymeBoundaryKernel.hpp>
#include <femx/assembly/EnzymeVolumeKernel.hpp>
#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/BoundaryElementValues.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/Quadrature.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/linalg/DenseLinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/backends/native/DenseSystemMatrix.hpp>
#include <femx/opt/TaoOptimizer.hpp>
#include <femx/problem/MatrixResidual.hpp>
#include <femx/problem/ObjectiveFunctional.hpp>
#include <femx/problem/SumObjective.hpp>
#include <femx/state/AdjointReducedFunctional.hpp>
#include <femx/state/MatrixAdjointSolver.hpp>
#include <femx/state/MatrixLinearStateSolver.hpp>
#include <femx/state/StateSolver.hpp>

namespace
{

using namespace femx;
using namespace femx::assembly;
using namespace femx::linalg;
using namespace femx::opt;
using namespace femx::problem;
using namespace femx::state;

using Dofs = Vector<Index>;

constexpr Index kLeftTag     = 1;
constexpr Index kRightTag    = 2;
constexpr Index kBottomTag   = 3;
constexpr Index kTopTag      = 4;
constexpr Index kVizInterval = 5;
constexpr Real  kPi          = 3.141592653589793238462643383279502884;

struct VisualizationOptions
{
  bool        enabled  = true;
  Index       interval = kVizInterval;
  std::string base     = "poisson-boundary-neumann-history";
};

void checkPetsc(PetscErrorCode ierr, const std::string& action)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        action + " failed with PETSc error code " + std::to_string(ierr));
  }
}

VisualizationOptions readVisualizationOptions()
{
  VisualizationOptions opts;
#ifndef FEMX_HAS_HDF5
  opts.enabled = false;
#endif

  PetscBool viz_enabled = opts.enabled ? PETSC_TRUE : PETSC_FALSE;
  checkPetsc(PetscOptionsGetBool(
                 nullptr, nullptr, "-viz", &viz_enabled, nullptr),
             "PetscOptionsGetBool(-viz)");

  opts.enabled = viz_enabled == PETSC_TRUE;
#ifndef FEMX_HAS_HDF5
  opts.enabled = false;
#endif

  PetscInt interval = static_cast<PetscInt>(opts.interval);
  checkPetsc(PetscOptionsGetInt(
                 nullptr, nullptr, "-viz_interval", &interval, nullptr),
             "PetscOptionsGetInt(-viz_interval)");
  opts.interval = static_cast<Index>(interval);
  if (opts.enabled && opts.interval <= 0)
  {
    throw std::runtime_error("-viz_interval must be positive");
  }

  char      base[4096]   = {};
  PetscBool basename_set = PETSC_FALSE;
  checkPetsc(PetscOptionsGetString(nullptr,
                                   nullptr,
                                   "-viz_output",
                                   base,
                                   sizeof(base),
                                   &basename_set),
             "PetscOptionsGetString(-viz_output)");
  if (basename_set == PETSC_TRUE)
  {
    opts.base = base;
  }

  return opts;
}

void addBoundaryFacet(Mesh&              mesh,
                      Index              tag,
                      const std::string& name,
                      Vector<Index>      nids)
{
  Mesh::BoundaryFacet facet;
  facet.dim        = 1;
  facet.entity_tag = tag;
  facet.ptag       = tag;
  facet.pname      = name;
  facet.shape      = Cell::Shape::Segment;
  facet.nids       = std::move(nids);
  mesh.addBoundaryFacet(std::move(facet));
}

Mesh makeUnitSquareMesh(Index nx, Index ny)
{
  Mesh mesh = Mesh::makeStructuredQuad(nx, ny);

  mesh.addPhysicalName(1, kLeftTag, "left");
  mesh.addPhysicalName(1, kRightTag, "right");
  mesh.addPhysicalName(1, kBottomTag, "bottom");
  mesh.addPhysicalName(1, kTopTag, "top");

  const Index nodes_per_row = nx + 1;

  for (Index i = 0; i < nx; ++i)
  {
    addBoundaryFacet(mesh, kBottomTag, "bottom", {i, i + 1});
  }

  for (Index j = 0; j < ny; ++j)
  {
    const Index n0 = j * nodes_per_row + nx;
    const Index n1 = (j + 1) * nodes_per_row + nx;
    addBoundaryFacet(mesh, kRightTag, "right", {n0, n1});
  }

  for (Index i = nx; i-- > 0;)
  {
    const Index n0 = ny * nodes_per_row + i + 1;
    const Index n1 = ny * nodes_per_row + i;
    addBoundaryFacet(mesh, kTopTag, "top", {n0, n1});
  }

  for (Index j = ny; j-- > 0;)
  {
    const Index n0 = j * nodes_per_row;
    const Index n1 = (j - 1) * nodes_per_row;
    addBoundaryFacet(mesh, kLeftTag, "left", {n0, n1});
  }

  return mesh;
}

Dofs makeDirichletDofs(const FESpace& space)
{
  constexpr Real eps = 1.0e-14;

  Vector<Real> constrained(space.numDofs());
  for (Index in = 0; in < space.mesh().numNodes(); ++in)
  {
    const auto& x = space.mesh().node(in);
    if (std::abs(x[0]) < eps || std::abs(x[0] - 1.0) < eps
        || std::abs(x[1]) < eps)
    {
      constrained[space.globalDof(in, 0)] = 1.0;
    }
  }

  Dofs dofs;
  for (Index i = 0; i < space.numDofs(); ++i)
  {
    if (constrained[i] != 0.0)
    {
      dofs.push_back(i);
    }
  }
  return dofs;
}

void getElementValues(const FESpace&      space,
                      const Vector<Real>& global,
                      Index               ic,
                      Dofs&               dofs,
                      Vector<Real>&       local)
{
  space.elemDofs(ic, dofs);
  const Index num_local_dofs = dofs.size();

  resizeOrZero(local, num_local_dofs);
  for (Index i = 0; i < local.size(); ++i)
  {
    local[i] = global[dofs[i]];
  }
}

void poissonVolumeResidual(Index       cell,
                           Index       nq,
                           Index       nn,
                           Index       dim,
                           Index       nres,
                           Index       nst,
                           Index       nprm,
                           const Real* N,
                           const Real* dNdx,
                           const Real* JxW,
                           const Real* state,
                           const Real* prm,
                           Real*       out)
{
  (void) cell;
  (void) nn;
  (void) nprm;
  (void) N;
  (void) prm;

  for (Index a = 0; a < nres; ++a)
  {
    out[a] = 0.0;
  }

  for (Index iq = 0; iq < nq; ++iq)
  {
    const Real* dNdx_q = dNdx + iq * nst * dim;

    Real grad_u[3] = {0.0, 0.0, 0.0};
    for (Index b = 0; b < nst; ++b)
    {
      for (Index d = 0; d < dim; ++d)
      {
        grad_u[d] += state[b] * dNdx_q[b * dim + d];
      }
    }

    for (Index a = 0; a < nres; ++a)
    {
      Real term = 0.0;
      for (Index d = 0; d < dim; ++d)
      {
        term += dNdx_q[a * dim + d] * grad_u[d];
      }
      out[a] += term * JxW[iq];
    }
  }
}

class PoissonVolumeEquation final : public MatrixResidual
{
public:
  PoissonVolumeEquation(const FESpace&         space,
                        Index                  nprm,
                        const GaussQuadrature& quad)
    : space_(space),
      nprm_(nprm),
      kernel_(space,
              quad,
              space.numDofsPerElem(),
              space.numDofsPerElem(),
              0)
  {
  }

  Index numStates() const override
  {
    return space_.numDofs();
  }

  Index numParams() const override
  {
    return nprm_;
  }

  Index numRes() const override
  {
    return space_.numDofs();
  }

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    checkSizes(state, prm);
    resizeOrZero(out, numRes());

    Dofs         dofs;
    Vector<Real> state_e;
    Vector<Real> prm_e;
    Vector<Real> res_e(space_.numDofsPerElem());

    for (Index ic = 0; ic < space_.numElems(); ++ic)
    {
      getElementValues(space_, state, ic, dofs, state_e);
      kernel_.res(ic, state_e, prm_e, res_e);

      for (Index a = 0; a < res_e.size(); ++a)
      {
        out[dofs[a]] += res_e[a];
      }
    }
  }

  void assembleStateJac(const Vector<Real>& state,
                        const Vector<Real>& prm,
                        SystemMatrix&       out) const override
  {
    checkSizes(state, prm);
    out.resize(numRes(), numStates());
    out.setZero();

    Dofs         dofs;
    Vector<Real> state_e;
    Vector<Real> prm_e;
    DenseMatrix  jac_e;

    for (Index ic = 0; ic < space_.numElems(); ++ic)
    {
      getElementValues(space_, state, ic, dofs, state_e);
      kernel_.stateJac(ic, state_e, prm_e, jac_e);
      for (Index a = 0; a < jac_e.rows(); ++a)
      {
        for (Index b = 0; b < jac_e.cols(); ++b)
        {
          out.add(dofs[a], dofs[b], jac_e(a, b));
        }
      }
    }
  }

  void assembleParamJac(const Vector<Real>& state,
                        const Vector<Real>& prm,
                        SystemMatrix&       out) const override
  {
    checkSizes(state, prm);
    out.resize(numRes(), numParams());
    out.setZero();
  }

private:
  void checkSizes(const Vector<Real>& state,
                  const Vector<Real>& prm) const
  {
    if (state.size() != numStates() || prm.size() != numParams())
    {
      throw std::runtime_error("PoissonVolumeEquation size mismatch");
    }
  }

private:
  const FESpace&                            space_;
  Index                                     nprm_{0};
  EnzymeVolumeKernel<poissonVolumeResidual> kernel_;
};

class DirichletResidualEquation final : public MatrixResidual
{
public:
  DirichletResidualEquation(
      const MatrixResidual& base_problem,
      Dofs                  dofs)
    : base_problem_(base_problem),
      dofs_(std::move(dofs))
  {
  }

  Index numStates() const override
  {
    return base_problem_.numStates();
  }

  Index numParams() const override
  {
    return base_problem_.numParams();
  }

  Index numRes() const override
  {
    return base_problem_.numRes();
  }

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    base_problem_.res(state, prm, out);
    for (Index dof : dofs_)
    {
      out[dof] = state[dof];
    }
  }

  void assembleStateJac(const Vector<Real>& state,
                        const Vector<Real>& prm,
                        SystemMatrix&       out) const override
  {
    base_problem_.assembleStateJac(state, prm, out);
    for (Index row : dofs_)
    {
      for (Index col = 0; col < out.numCols(); ++col)
      {
        out.set(row, col, 0.0);
      }
      out.set(row, row, 1.0);
    }
  }

  void assembleParamJac(const Vector<Real>& state,
                        const Vector<Real>& prm,
                        SystemMatrix&       out) const override
  {
    base_problem_.assembleParamJac(state, prm, out);
    for (Index row : dofs_)
    {
      for (Index col = 0; col < out.numCols(); ++col)
      {
        out.set(row, col, 0.0);
      }
    }
  }

private:
  const MatrixResidual& base_problem_;
  Dofs                  dofs_;
};

void boundaryNeumannResidual(Index       facet,
                             Index       nq,
                             Index       nn,
                             Index       dim,
                             Index       nres,
                             Index       nst,
                             Index       nprm,
                             const Real* N,
                             const Real* point,
                             const Real* nrm,
                             const Real* JxW,
                             const Real* state,
                             const Real* prm,
                             Real*       out)
{
  (void) facet;
  (void) dim;
  (void) nres;
  (void) nst;
  (void) nprm;
  (void) point;
  (void) nrm;
  (void) state;

  for (Index a = 0; a < nn; ++a)
  {
    out[a] = 0.0;
  }

  for (Index iq = 0; iq < nq; ++iq)
  {
    const Real* Nq = N + iq * nn;

    Real mq = 0.0;
    for (Index b = 0; b < nn; ++b)
    {
      mq += Nq[b] * prm[b];
    }

    for (Index a = 0; a < nn; ++a)
    {
      out[a] -= Nq[a] * mq * JxW[iq];
    }
  }
}

class VelocityDataMisfit final : public ObjectiveFunctional
{
public:
  VelocityDataMisfit(const FESpace&         space,
                     Index                  nprm,
                     const Vector<Real>&    data,
                     const GaussQuadrature& quad,
                     Real                   wt)
    : space_(space),
      nprm_(nprm),
      data_(data),
      quad_(quad),
      weight_(wt)
  {
    if (data_.size() != space_.numDofs() || weight_ < 0.0)
    {
      throw std::runtime_error(
          "VelocityDataMisfit received inconsistent data or weight");
    }
  }

  Index numStates() const override
  {
    return space_.numDofs();
  }

  Index numParams() const override
  {
    return nprm_;
  }

  Real value(const Vector<Real>& state,
             const Vector<Real>& prm) const override
  {
    checkSizes(state, prm);

    return integrateStateDifference(state, nullptr);
  }

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    checkSizes(state, prm);
    resizeOrZero(out, numStates());

    (void) integrateStateDifference(state, &out);
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    checkSizes(state, prm);
    resizeOrZero(out, numParams());
  }

private:
  Real integrateStateDifference(const Vector<Real>& state,
                                Vector<Real>*       state_grad) const
  {
    ElementValues vals(space_.finiteElement(), quad_);
    Dofs          dofs;
    Real          value_out = 0.0;

    for (Index ic = 0; ic < space_.numElems(); ++ic)
    {
      space_.elemDofs(ic, dofs);
      vals.reinit(space_.mesh().cell(ic));

      for (Index iq = 0; iq < vals.numQuadraturePoints(); ++iq)
      {
        const auto N = vals.N(iq);

        Real diff_q = 0.0;
        for (Index a = 0; a < vals.numDofs(); ++a)
        {
          const Index dof  = dofs[a];
          diff_q          += N[a] * (state[dof] - data_[dof]);
        }

        const Real w  = weight_ * vals.JxW(iq);
        value_out    += 0.5 * diff_q * diff_q * w;

        if (state_grad != nullptr)
        {
          for (Index a = 0; a < vals.numDofs(); ++a)
          {
            const Index dof     = dofs[a];
            (*state_grad)[dof] += N[a] * diff_q * w;
          }
        }
      }
    }
    return value_out;
  }

  void checkSizes(const Vector<Real>& state,
                  const Vector<Real>& prm) const
  {
    if (state.size() != numStates() || prm.size() != numParams())
    {
      throw std::runtime_error("VelocityDataMisfit size mismatch");
    }
  }

private:
  const FESpace&  space_;
  Index           nprm_{0};
  Vector<Real>    data_;
  GaussQuadrature quad_;
  Real            weight_{1.0};
};

class BoundaryRegularization final : public ObjectiveFunctional
{
public:
  BoundaryRegularization(
      const Mesh&            mesh,
      BoundaryDofLayout      param_layout,
      Index                  nst,
      const GaussQuadrature& quad,
      Real                   wt)
    : mesh_(mesh),
      param_layout_(std::move(param_layout)),
      nst_(nst),
      quad_(quad),
      weight_(wt)
  {
    if (weight_ < 0.0)
    {
      throw std::runtime_error(
          "BoundaryRegularization received negative weight");
    }
  }

  Index numStates() const override
  {
    return nst_;
  }

  Index numParams() const override
  {
    return param_layout_.numDofs();
  }

  Real value(const Vector<Real>& state,
             const Vector<Real>& prm) const override
  {
    checkSizes(state, prm);

    return integrateBoundaryL2(prm, nullptr);
  }

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    checkSizes(state, prm);
    resizeOrZero(out, numStates());
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    checkSizes(state, prm);
    resizeOrZero(out, numParams());

    (void) integrateBoundaryL2(prm, &out);
  }

private:
  Real integrateBoundaryL2(const Vector<Real>& prm,
                           Vector<Real>*       param_grad) const
  {
    BoundaryElementValues vals(quad_);
    Dofs                  dofs;
    Real                  value_out = 0.0;

    for (Index ib = 0; ib < param_layout_.numFacets(); ++ib)
    {
      param_layout_.facetDofs(ib, dofs);
      vals.reinit(mesh_, param_layout_.facet(ib));

      for (Index iq = 0; iq < vals.numQuadraturePoints(); ++iq)
      {
        const auto N = vals.N(iq);

        Real mq = 0.0;
        for (Index a = 0; a < vals.numNodes(); ++a)
        {
          mq += N[a] * prm[dofs[a]];
        }

        const Real w  = weight_ * vals.JxW(iq);
        value_out    += 0.5 * mq * mq * w;

        if (param_grad != nullptr)
        {
          for (Index a = 0; a < vals.numNodes(); ++a)
          {
            (*param_grad)[dofs[a]] += N[a] * mq * w;
          }
        }
      }
    }
    return value_out;
  }

  void checkSizes(const Vector<Real>& state,
                  const Vector<Real>& prm) const
  {
    if (state.size() != numStates() || prm.size() != numParams())
    {
      throw std::runtime_error("BoundaryRegularization size mismatch");
    }
  }

private:
  const Mesh&       mesh_;
  BoundaryDofLayout param_layout_;
  Index             nst_{0};
  GaussQuadrature   quad_;
  Real              weight_{0.0};
};

Vector<Real> boundaryCoordinates(const Mesh&              mesh,
                                 const BoundaryDofLayout& lyt)
{
  Vector<Real> x(lyt.numDofs());
  Vector<Real> seen(lyt.numDofs());
  Dofs         dofs;

  for (Index ib = 0; ib < lyt.numFacets(); ++ib)
  {
    lyt.facetDofs(ib, dofs);
    const auto& facet          = lyt.facet(ib);
    const Index num_facet_dofs = dofs.size();

    for (Index a = 0; a < num_facet_dofs; ++a)
    {
      const Index dof = dofs[a];
      if (seen[dof] == 0.0)
      {
        const Index node = facet.nids[a];
        x[dof]           = mesh.node(node)[0];
        seen[dof]        = 1.0;
      }
    }
  }
  return x;
}

Vector<Real> makeTrueNeumann(const Vector<Real>& x)
{
  Vector<Real> neumann(x.size());
  for (Index i = 0; i < neumann.size(); ++i)
  {
    neumann[i] = std::sin(2.0 * kPi * x[i]);
  }
  return neumann;
}

Vector<Real> boundaryFieldOnMesh(const Mesh&              mesh,
                                 const BoundaryDofLayout& lyt,
                                 const Vector<Real>&      vals)
{
  if (vals.size() != lyt.numDofs())
  {
    throw std::runtime_error("boundaryFieldOnMesh size mismatch");
  }

  Vector<Real> field(mesh.numNodes());
  Dofs         dofs;
  for (Index ib = 0; ib < lyt.numFacets(); ++ib)
  {
    lyt.facetDofs(ib, dofs);
    const auto& facet = lyt.facet(ib);
    Index       a     = 0;
    for (Index node : facet.nids)
    {
      field[node] = vals[dofs[a]];
      ++a;
    }
  }
  return field;
}

class OptimizationHistoryOutput
{
public:
  OptimizationHistoryOutput(const Mesh&              mesh,
                            const BoundaryDofLayout& param_layout,
                            StateSolver&             state_solver,
                            const Vector<Real>&      observed_state,
                            const Vector<Real>&      true_prm,
                            VisualizationOptions     opts)
    : mesh_(mesh),
      param_layout_(param_layout),
      state_solver_(state_solver),
      observed_state_(observed_state),
      true_prm_(true_prm),
      true_param_field_(boundaryFieldOnMesh(mesh, param_layout, true_prm)),
      opts_(std::move(opts))
  {
    if (opts_.enabled)
    {
      out_.attachMesh(mesh_);
    }
  }

  const VisualizationOptions& opts() const
  {
    return opts_;
  }

  bool hasRecorded() const
  {
    return recorded_;
  }

  void rec(Index iter, const Vector<Real>& prm, bool force = false)
  {
    if (!opts_.enabled)
    {
      return;
    }
    if (!force && iter % opts_.interval != 0)
    {
      return;
    }
    if (recorded_ && last_iter_ == iter)
    {
      return;
    }

    Vector<Real> state;
    state_solver_.solve(prm, state);

    out_.beginStep(static_cast<Real>(iter));
    out_.addNodalScalarField("u", state);
    out_.addNodalScalarField("u_obs", observed_state_);
    out_.addNodalScalarField("u_error", difference(state, observed_state_));

    out_.addNodalScalarField("m", boundaryField(prm));
    out_.addNodalScalarField("m_true", true_param_field_);

    const auto param_error = difference(prm, true_prm_);
    out_.addNodalScalarField("m_error", boundaryField(param_error));

    recorded_  = true;
    last_iter_ = iter;
  }

  void write() const
  {
    if (opts_.enabled && recorded_)
    {
      out_.write(opts_.base);
    }
  }

private:
  Vector<Real> boundaryField(const Vector<Real>& vals) const
  {
    return boundaryFieldOnMesh(mesh_, param_layout_, vals);
  }

  const Mesh&              mesh_;
  const BoundaryDofLayout& param_layout_;
  StateSolver&             state_solver_;
  const Vector<Real>&      observed_state_;
  const Vector<Real>&      true_prm_;
  Vector<Real>             true_param_field_;
  VisualizationOptions     opts_;
  TimeSeriesDataOut        out_;
  bool                     recorded_{false};
  Index                    last_iter_{-1};
};

Real DirectionalDerivative(
    AdjointReducedFunctional& fn,
    const Vector<Real>&       prm,
    const Vector<Real>&       dir,
    Real                      eps)
{
  Vector<Real> plus(prm.size());
  Vector<Real> minus(prm.size());
  for (Index i = 0; i < prm.size(); ++i)
  {
    plus[i]  = prm[i] + eps * dir[i];
    minus[i] = prm[i] - eps * dir[i];
  }
  return (fn.value(plus) - fn.value(minus)) / (2.0 * eps);
}

int run()
{
  const Index nx   = 8;
  const Index ny   = 8;
  const Real  beta = 1.0e-6;

  Mesh mesh = makeUnitSquareMesh(nx, ny);

  LagrangeQuadQ1 elem;
  FESpace        space(&mesh, &elem);
  space.setup();

  const auto vol_quad = GaussQuadrature::quadrilateral(2);
  const auto bd_quad  = GaussQuadrature::segment(2);

  BoundaryDofLayout top_state_layout(space, kTopTag);
  BoundaryDofLayout top_param_layout = BoundaryDofLayout::compact(space, kTopTag);

  PoissonVolumeEquation vol_eq(
      space, top_param_layout.numDofs(), vol_quad);

  EnzymeBoundaryKernel<boundaryNeumannResidual>
      neumann_kernel(mesh, bd_quad, 2, 2, 2);

  BoundaryResidualEquation neumann_problem(
      vol_eq,
      top_state_layout,
      top_state_layout,
      top_param_layout,
      neumann_kernel);

  DirichletResidualEquation problem(neumann_problem, makeDirichletDofs(space));

  DenseSystemMatrix       state_jac;
  DenseLinearSolver       state_lin_solver;
  MatrixLinearStateSolver state_solver(problem, state_jac, state_lin_solver);

  const auto   top_x        = boundaryCoordinates(mesh, top_param_layout);
  Vector<Real> neumann_true = makeTrueNeumann(top_x);

  Vector<Real> u_obs;
  state_solver.solve(neumann_true, u_obs);

  VelocityDataMisfit misfit(
      space,
      top_param_layout.numDofs(),
      u_obs,
      vol_quad,
      1.0);
  BoundaryRegularization reg(
      mesh,
      top_param_layout,
      space.numDofs(),
      bd_quad,
      beta);

  SumObjective obj(space.numDofs(), top_param_layout.numDofs());
  obj.add(misfit).add(reg);

  DenseSystemMatrix   adj_jac;
  DenseLinearSolver   adj_lin_solver;
  MatrixAdjointSolver adj_solver(problem, adj_jac, adj_lin_solver);

  AdjointReducedFunctional reduced(state_solver, adj_solver, problem, obj);

  Vector<Real> initial(top_param_layout.numDofs());
  initial.setZero();

  const VisualizationOptions viz_options = readVisualizationOptions();
  OptimizationHistoryOutput  hist(
      mesh,
      top_param_layout,
      state_solver,
      u_obs,
      neumann_true,
      viz_options);
  hist.rec(0, initial, true);

  Vector<Real> initial_grad;
  const Real   initial_value = reduced.valueGrad(initial, initial_grad);

  Vector<Real> dir(top_param_layout.numDofs());
  for (Index i = 0; i < dir.size(); ++i)
  {
    dir[i] = std::cos(0.73 * (i + 1));
  }
  const Real dir_norm = norm(dir);
  for (Index i = 0; i < dir.size(); ++i)
  {
    dir[i] /= dir_norm;
  }

  const Real adjoint_dir = dot(initial_grad, dir);
  const Real fd_dir      = DirectionalDerivative(reduced, initial, dir, 1.0e-6);

  TaoOptimizer optimizer(reduced);
  optimizer.opts().type     = TAOLMVM;
  optimizer.opts().abs_tol  = 1.0e-10;
  optimizer.opts().rel_tol  = 1.0e-10;
  optimizer.opts().step_tol = 0.0;
  optimizer.opts().max_its  = 160;

  optimizer.setMonitor(
      [&hist](const TaoIterationInfo& info, const Vector<Real>& prm)
      {
        hist.rec(info.its, prm);
      });

  TaoResult            result;
  const PetscErrorCode ierr = optimizer.solve(initial, result);
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        "TAO solve failed with PETSc error code "
        + std::to_string(ierr));
  }

  hist.rec(result.its, result.prm, true);
  hist.write();

  const Real final_rmse = rmse(result.prm, neumann_true);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "2D Poisson boundary Neumann inverse problem\n";
  std::cout << "  mesh: " << nx << " x " << ny << " Q1 cells\n";
  std::cout << "  states: " << space.numDofs()
            << ", top Neumann parameters: " << top_param_layout.numDofs()
            << '\n';
  std::cout << "  beta: " << beta << '\n';
  std::cout << "  initial objective: " << initial_value << '\n';
  std::cout << "  derivative check: adjoint = " << adjoint_dir
            << ", finite difference = " << fd_dir
            << ", error = " << std::abs(adjoint_dir - fd_dir) << '\n';
  std::cout << "  TAO converged: " << (result.converged() ? "yes" : "no")
            << ", reason = " << result.reason
            << ", iterations = " << result.its << '\n';
  std::cout << "  final objective: " << result.value
            << ", |grad| = " << std::sqrt(result.grad_norm_squared)
            << ", Neumann RMSE = " << final_rmse << '\n';
  if (hist.opts().enabled && hist.hasRecorded())
  {
    std::cout << "  visualization: " << hist.opts().base
              << ".xdmf/.h5, interval = "
              << hist.opts().interval << '\n';
  }
  else
  {
    std::cout << "  visualization: disabled";
#ifndef FEMX_HAS_HDF5
    std::cout << " (HDF5 support not available)";
#endif
    std::cout << '\n';
  }

  std::cout << "\n  x            neumann_true neumann_opt\n";
  for (Index i = 0; i < result.prm.size(); ++i)
  {
    std::cout << "  " << std::setw(10)
              << top_x[i] << "  "
              << std::setw(10) << neumann_true[i] << "  "
              << std::setw(10) << result.prm[i] << '\n';
  }

  return result.converged() ? 0 : 2;
}

} // namespace

int main(int argc, char** argv)
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (ierr != PETSC_SUCCESS)
  {
    return 1;
  }

  int exit_code = 0;
  try
  {
    exit_code = run();
  }
  catch (const std::exception& e)
  {
    std::cerr << "poisson-boundary-neumann-tao failed: " << e.what() << '\n';
    exit_code = 1;
  }

  ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS && exit_code == 0)
  {
    return 1;
  }
  return exit_code;
}
