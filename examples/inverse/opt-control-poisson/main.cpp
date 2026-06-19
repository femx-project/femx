#include <petsctao.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <femx/assembly/BoundaryDofLayout.hpp>
#include <femx/assembly/BoundaryResidualEquation.hpp>
#include <femx/assembly/EnzymeBoundaryKernel.hpp>
#include <femx/assembly/EnzymeVolumeKernel.hpp>
#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/eq/MatrixLinearStateSolver.hpp>
#include <femx/eq/MatrixResidualEquation.hpp>
#include <femx/fem/BoundaryElementValues.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/inverse/AdjointReducedFunctional.hpp>
#include <femx/inverse/MatrixAdjointSolver.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/inverse/SumObjectiveFunctional.hpp>
#include <femx/inverse/petsc/TaoOptimizer.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/DenseLinearSolver.hpp>
#include <femx/system/native/DenseSystemMatrix.hpp>

namespace
{

using namespace femx;
using namespace femx::assembly;
using namespace femx::eq;
using namespace femx::inverse;
using namespace femx::system;

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
  std::string basename = "poisson-boundary-neumann-history";
};

void checkPetsc(PetscErrorCode ierr, const std::string& action)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        action + " failed with PETSc error code " + std::to_string(ierr));
  }
}

void resize(Vector<Real>& out, Index size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

VisualizationOptions readVisualizationOptions()
{
  VisualizationOptions options;
#ifndef FEMX_HAS_HDF5
  options.enabled = false;
#endif

  PetscBool viz_enabled = options.enabled ? PETSC_TRUE : PETSC_FALSE;
  checkPetsc(PetscOptionsGetBool(
                 nullptr, nullptr, "-viz", &viz_enabled, nullptr),
             "PetscOptionsGetBool(-viz)");

  options.enabled = viz_enabled == PETSC_TRUE;
#ifndef FEMX_HAS_HDF5
  options.enabled = false;
#endif

  PetscInt interval = static_cast<PetscInt>(options.interval);
  checkPetsc(PetscOptionsGetInt(
                 nullptr, nullptr, "-viz_interval", &interval, nullptr),
             "PetscOptionsGetInt(-viz_interval)");
  options.interval = static_cast<Index>(interval);
  if (options.enabled && options.interval <= 0)
  {
    throw std::runtime_error("-viz_interval must be positive");
  }

  char      basename[4096] = {};
  PetscBool basename_set   = PETSC_FALSE;
  checkPetsc(PetscOptionsGetString(nullptr,
                                   nullptr,
                                   "-viz_output",
                                   basename,
                                   sizeof(basename),
                                   &basename_set),
             "PetscOptionsGetString(-viz_output)");
  if (basename_set == PETSC_TRUE)
  {
    options.basename = basename;
  }

  return options;
}

void addBoundaryFacet(Mesh&              mesh,
                      Index              tag,
                      const std::string& name,
                      Dofs               node_ids)
{
  Mesh::BoundaryFacet facet;
  facet.dim           = 1;
  facet.entity_tag    = tag;
  facet.physical_tag  = tag;
  facet.physical_name = name;
  facet.shape         = Cell::Shape::Segment;
  facet.node_ids      = std::move(node_ids);
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

  resize(local, num_local_dofs);
  for (Index i = 0; i < local.size(); ++i)
  {
    local[i] = global[dofs[i]];
  }
}

void poissonVolumeResidual(Index       cell,
                           Index       num_qp,
                           Index       num_nodes,
                           Index       dim,
                           Index       num_res,
                           Index       num_states,
                           Index       num_prm,
                           const Real* N,
                           const Real* dNdx,
                           const Real* JxW,
                           const Real* state,
                           const Real* prm,
                           Real*       out)
{
  (void) cell;
  (void) num_nodes;
  (void) num_prm;
  (void) N;
  (void) prm;

  for (Index a = 0; a < num_res; ++a)
  {
    out[a] = 0.0;
  }

  for (Index iq = 0; iq < num_qp; ++iq)
  {
    const Real* dNdx_q = dNdx + iq * num_states * dim;

    Real grad_u[3] = {0.0, 0.0, 0.0};
    for (Index b = 0; b < num_states; ++b)
    {
      for (Index d = 0; d < dim; ++d)
      {
        grad_u[d] += state[b] * dNdx_q[b * dim + d];
      }
    }

    for (Index a = 0; a < num_res; ++a)
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

class PoissonVolumeEquation final : public MatrixResidualEquation
{
public:
  PoissonVolumeEquation(const FESpace&         space,
                        Index                  num_prm,
                        const GaussQuadrature& quad)
    : space_(space),
      num_prm_(num_prm),
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
    return num_prm_;
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
    resize(out, numRes());

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
  Index                                     num_prm_{0};
  EnzymeVolumeKernel<poissonVolumeResidual> kernel_;
};

class DirichletResidualEquation final : public MatrixResidualEquation
{
public:
  DirichletResidualEquation(
      const MatrixResidualEquation& base_eq,
      Dofs                          dofs)
    : base_eq_(base_eq),
      dofs_(std::move(dofs))
  {
  }

  Index numStates() const override
  {
    return base_eq_.numStates();
  }

  Index numParams() const override
  {
    return base_eq_.numParams();
  }

  Index numRes() const override
  {
    return base_eq_.numRes();
  }

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    base_eq_.res(state, prm, out);
    for (Index dof : dofs_)
    {
      out[dof] = state[dof];
    }
  }

  void assembleStateJac(const Vector<Real>& state,
                        const Vector<Real>& prm,
                        SystemMatrix&       out) const override
  {
    base_eq_.assembleStateJac(state, prm, out);
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
    base_eq_.assembleParamJac(state, prm, out);
    for (Index row : dofs_)
    {
      for (Index col = 0; col < out.numCols(); ++col)
      {
        out.set(row, col, 0.0);
      }
    }
  }

private:
  const MatrixResidualEquation& base_eq_;
  Dofs                          dofs_;
};

void boundaryNeumannResidual(Index       facet,
                             Index       num_qp,
                             Index       num_nodes,
                             Index       dim,
                             Index       num_res,
                             Index       num_states,
                             Index       num_prm,
                             const Real* N,
                             const Real* point,
                             const Real* normal,
                             const Real* JxW,
                             const Real* state,
                             const Real* prm,
                             Real*       out)
{
  (void) facet;
  (void) dim;
  (void) num_res;
  (void) num_states;
  (void) num_prm;
  (void) point;
  (void) normal;
  (void) state;

  for (Index a = 0; a < num_nodes; ++a)
  {
    out[a] = 0.0;
  }

  for (Index iq = 0; iq < num_qp; ++iq)
  {
    const Real* Nq = N + iq * num_nodes;

    Real mq = 0.0;
    for (Index b = 0; b < num_nodes; ++b)
    {
      mq += Nq[b] * prm[b];
    }

    for (Index a = 0; a < num_nodes; ++a)
    {
      out[a] -= Nq[a] * mq * JxW[iq];
    }
  }
}

class VelocityDataMisfit final : public ObjectiveFunctional
{
public:
  VelocityDataMisfit(const FESpace&         space,
                     Index                  num_prm,
                     const Vector<Real>&    data,
                     const GaussQuadrature& quad,
                     Real                   weight)
    : space_(space),
      num_prm_(num_prm),
      data_(data),
      quad_(quad),
      weight_(weight)
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
    return num_prm_;
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
    resize(out, numStates());

    (void) integrateStateDifference(state, &out);
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    checkSizes(state, prm);
    resize(out, numParams());
  }

private:
  Real integrateStateDifference(const Vector<Real>& state,
                                Vector<Real>*       state_grad) const
  {
    ElementValues values(space_.finiteElement(), quad_);
    Dofs          dofs;
    Real          value_out = 0.0;

    for (Index ic = 0; ic < space_.numElems(); ++ic)
    {
      space_.elemDofs(ic, dofs);
      values.reinit(space_.mesh().cell(ic));

      for (Index iq = 0; iq < values.numQuadraturePoints(); ++iq)
      {
        const auto N = values.N(iq);

        Real diff_q = 0.0;
        for (Index a = 0; a < values.numDofs(); ++a)
        {
          const Index dof  = dofs[a];
          diff_q          += N[a] * (state[dof] - data_[dof]);
        }

        const Real w  = weight_ * values.JxW(iq);
        value_out    += 0.5 * diff_q * diff_q * w;

        if (state_grad != nullptr)
        {
          for (Index a = 0; a < values.numDofs(); ++a)
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
  Index           num_prm_{0};
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
      Index                  num_states,
      const GaussQuadrature& quad,
      Real                   weight)
    : mesh_(mesh),
      param_layout_(std::move(param_layout)),
      num_states_(num_states),
      quad_(quad),
      weight_(weight)
  {
    if (weight_ < 0.0)
    {
      throw std::runtime_error(
          "BoundaryRegularization received negative weight");
    }
  }

  Index numStates() const override
  {
    return num_states_;
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
    resize(out, numStates());
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    checkSizes(state, prm);
    resize(out, numParams());

    (void) integrateBoundaryL2(prm, &out);
  }

private:
  Real integrateBoundaryL2(const Vector<Real>& prm,
                           Vector<Real>*       param_grad) const
  {
    BoundaryElementValues values(quad_);
    Dofs                  dofs;
    Real                  value_out = 0.0;

    for (Index ib = 0; ib < param_layout_.numFacets(); ++ib)
    {
      param_layout_.facetDofs(ib, dofs);
      values.reinit(mesh_, param_layout_.facet(ib));

      for (Index iq = 0; iq < values.numQuadraturePoints(); ++iq)
      {
        const auto N = values.N(iq);

        Real mq = 0.0;
        for (Index a = 0; a < values.numNodes(); ++a)
        {
          mq += N[a] * prm[dofs[a]];
        }

        const Real w  = weight_ * values.JxW(iq);
        value_out    += 0.5 * mq * mq * w;

        if (param_grad != nullptr)
        {
          for (Index a = 0; a < values.numNodes(); ++a)
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
  Index             num_states_{0};
  GaussQuadrature   quad_;
  Real              weight_{0.0};
};

Vector<Real> boundaryCoordinates(const Mesh&              mesh,
                                 const BoundaryDofLayout& layout)
{
  Vector<Real> x(layout.numDofs());
  Vector<Real> seen(layout.numDofs());
  Dofs         dofs;

  for (Index ib = 0; ib < layout.numFacets(); ++ib)
  {
    layout.facetDofs(ib, dofs);
    const auto& facet          = layout.facet(ib);
    const Index num_facet_dofs = dofs.size();

    for (Index a = 0; a < num_facet_dofs; ++a)
    {
      const Index dof = dofs[a];
      if (seen[dof] == 0.0)
      {
        const Index in = facet.node_ids[a];
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
                                 const BoundaryDofLayout& layout,
                                 const Vector<Real>&      values)
{
  if (values.size() != layout.numDofs())
  {
    throw std::runtime_error("boundaryFieldOnMesh size mismatch");
  }

  Vector<Real> field(mesh.numNodes());
  Dofs         dofs;
  for (Index ib = 0; ib < layout.numFacets(); ++ib)
  {
    layout.facetDofs(ib, dofs);
    const auto& facet = layout.facet(ib);
    Index       a     = 0;
    for (Index node : facet.node_ids)
    {
      field[in] = values[dofs[a]];
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
                            VisualizationOptions     options)
    : mesh_(mesh),
      param_layout_(param_layout),
      state_solver_(state_solver),
      observed_state_(observed_state),
      true_prm_(true_prm),
      true_param_field_(boundaryFieldOnMesh(mesh, param_layout, true_prm)),
      options_(std::move(options))
  {
    if (options_.enabled)
    {
      out_.attachMesh(mesh_);
    }
  }

  const VisualizationOptions& options() const
  {
    return options_;
  }

  bool hasRecorded() const
  {
    return recorded_;
  }

  void record(Index iter, const Vector<Real>& prm, bool force = false)
  {
    if (!options_.enabled)
    {
      return;
    }
    if (!force && iter % options_.interval != 0)
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
    if (options_.enabled && recorded_)
    {
      out_.write(options_.basename);
    }
  }

private:
  Vector<Real> boundaryField(const Vector<Real>& values) const
  {
    return boundaryFieldOnMesh(mesh_, param_layout_, values);
  }

  const Mesh&              mesh_;
  const BoundaryDofLayout& param_layout_;
  StateSolver&             state_solver_;
  const Vector<Real>&      observed_state_;
  const Vector<Real>&      true_prm_;
  Vector<Real>             true_param_field_;
  VisualizationOptions     options_;
  TimeSeriesDataOut        out_;
  bool                     recorded_{false};
  Index                    last_iter_{-1};
};

Real DirectionalDerivative(
    ReducedFunctional&  functional,
    const Vector<Real>& prm,
    const Vector<Real>& dir,
    Real                eps)
{
  Vector<Real> plus(prm.size());
  Vector<Real> minus(prm.size());
  for (Index i = 0; i < prm.size(); ++i)
  {
    plus[i]  = prm[i] + eps * dir[i];
    minus[i] = prm[i] - eps * dir[i];
  }
  return (functional.value(plus) - functional.value(minus)) / (2.0 * eps);
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

  BoundaryResidualEquation neumann_eq(
      vol_eq,
      top_state_layout,
      top_state_layout,
      top_param_layout,
      neumann_kernel);

  DirichletResidualEquation eq(neumann_eq, makeDirichletDofs(space));

  DenseSystemMatrix       state_jac;
  DenseLinearSolver       state_lin_solver;
  MatrixLinearStateSolver state_solver(eq, state_jac, state_lin_solver);

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

  SumObjectiveFunctional obj(space.numDofs(), top_param_layout.numDofs());
  obj.add(misfit).add(reg);

  DenseSystemMatrix   adj_jac;
  DenseLinearSolver   adj_lin_solver;
  MatrixAdjointSolver adj_solver(eq, adj_jac, adj_lin_solver);

  AdjointReducedFunctional reduced(state_solver, adj_solver, eq, obj);

  Vector<Real> initial(top_param_layout.numDofs());
  initial.setZero();

  const VisualizationOptions viz_options = readVisualizationOptions();
  OptimizationHistoryOutput  history(
      mesh,
      top_param_layout,
      state_solver,
      u_obs,
      m_true,
      viz_options);
  history.record(0, initial, true);

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
  optimizer.options().type                = TAOLMVM;
  optimizer.options().grad_abs_tolerance  = 1.0e-10;
  optimizer.options().grad_rel_tolerance  = 1.0e-10;
  optimizer.options().grad_step_tolerance = 0.0;
  optimizer.options().max_its             = 160;
  optimizer.options().use_opts_db         = true;

  optimizer.setMonitor(
      [&history](const TaoIterationInfo& info, const Vector<Real>& prm)
      {
        history.record(info.its, prm);
      });

  TaoResult            result;
  const PetscErrorCode ierr = optimizer.solve(initial, result);
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        "TAO solve failed with PETSc error code "
        + std::to_string(ierr));
  }

  history.record(result.its, result.prm, true);
  history.write();

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
  if (history.options().enabled && history.hasRecorded())
  {
    std::cout << "  visualization: " << history.options().basename
              << ".xdmf/.h5, interval = "
              << history.options().interval << '\n';
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
