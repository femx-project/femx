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
#include <femx/assembly/EnzymeBoundaryIntegralKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/eq/AssembledLinearStateSolver.hpp>
#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/fem/BoundaryElementValues.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/inverse/AdjointReducedFunctional.hpp>
#include <femx/inverse/MatrixEquationAdjointSolver.hpp>
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

using Dofs = std::vector<Index>;

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
  std::string basename = "poisson-boundary-flux-history";
};

void checkPetsc(PetscErrorCode ierr, const std::string& action)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        action + " failed with PETSc error code " + std::to_string(ierr));
  }
}

void resize(Vector& out, Index size)
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

Real dot(const Vector& x, const Vector& y)
{
  if (x.size() != y.size())
  {
    throw std::runtime_error("dot received incompatible vectors");
  }

  Real value = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    value += x[i] * y[i];
  }
  return value;
}

Real norm2(const Vector& x)
{
  return std::sqrt(dot(x, x));
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

  Vector constrained(space.numDofs());
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

void gatherCell(const FESpace& space,
                const Vector&  global,
                Index          ic,
                Dofs&          dofs,
                Vector&        local)
{
  space.elemDofs(ic, dofs);
  const Index num_local_dofs = dofs.size();
  resize(local, num_local_dofs);
  for (Index i = 0; i < local.size(); ++i)
  {
    local[i] = global[dofs[i]];
  }
}

class PoissonInteriorEquation final
  : public AssembledResidualEquation
{
public:
  PoissonInteriorEquation(const FESpace&         space,
                          Index                  num_params,
                          const GaussQuadrature& quad)
    : space_(space),
      num_params_(num_params),
      quad_(quad)
  {
  }

  Index numStates() const override
  {
    return space_.numDofs();
  }

  Index numParams() const override
  {
    return num_params_;
  }

  Index numRes() const override
  {
    return space_.numDofs();
  }

  void res(const Vector& state,
           const Vector& params,
           Vector&       out) const override
  {
    checkSizes(state, params);
    resize(out, numRes());

    ElementValues values(space_.finiteElement(), quad_);
    Dofs          dofs;
    Vector        state_e;
    Vector        res_e(space_.numDofsPerElem());

    for (Index ic = 0; ic < space_.numElems(); ++ic)
    {
      gatherCell(space_, state, ic, dofs, state_e);
      res_e.setZero();

      values.reinit(space_.mesh().cell(ic));
      for (Index iq = 0; iq < values.numQuadraturePoints(); ++iq)
      {
        const auto dNdx = values.dNdx(iq);

        Real grad_u[2] = {0.0, 0.0};
        for (Index b = 0; b < values.numDofs(); ++b)
        {
          for (Index d = 0; d < values.dim(); ++d)
          {
            grad_u[d] += state_e[b] * dNdx(b, d);
          }
        }

        for (Index a = 0; a < values.numDofs(); ++a)
        {
          Real grad_term = 0.0;
          for (Index d = 0; d < values.dim(); ++d)
          {
            grad_term += dNdx(a, d) * grad_u[d];
          }
          res_e[a] += grad_term * values.JxW(iq);
        }
      }

      for (Index a = 0; a < res_e.size(); ++a)
      {
        out[dofs[a]] += res_e[a];
      }
    }
  }

  void assembleStateJac(const Vector& state,
                        const Vector& params,
                        SystemMatrix& out) const override
  {
    checkSizes(state, params);
    out.resize(numRes(), numStates());
    out.setZero();

    ElementValues values(space_.finiteElement(), quad_);
    Dofs          dofs;

    for (Index ic = 0; ic < space_.numElems(); ++ic)
    {
      space_.elemDofs(ic, dofs);
      values.reinit(space_.mesh().cell(ic));

      for (Index iq = 0; iq < values.numQuadraturePoints(); ++iq)
      {
        const auto dNdx = values.dNdx(iq);
        for (Index a = 0; a < values.numDofs(); ++a)
        {
          for (Index b = 0; b < values.numDofs(); ++b)
          {
            Real entry = 0.0;
            for (Index d = 0; d < values.dim(); ++d)
            {
              entry += dNdx(a, d) * dNdx(b, d);
            }
            out.add(dofs[a],
                    dofs[b],
                    entry * values.JxW(iq));
          }
        }
      }
    }
  }

  void assembleParamJac(const Vector& state,
                        const Vector& params,
                        SystemMatrix& out) const override
  {
    checkSizes(state, params);
    out.resize(numRes(), numParams());
    out.setZero();
  }

private:
  void checkSizes(const Vector& state,
                  const Vector& params) const
  {
    if (state.size() != numStates() || params.size() != numParams())
    {
      throw std::runtime_error("PoissonInteriorEquation size mismatch");
    }
  }

private:
  const FESpace&  space_;
  Index           num_params_{0};
  GaussQuadrature quad_;
};

class DirichletResidualEquation final
  : public AssembledResidualEquation
{
public:
  DirichletResidualEquation(
      const AssembledResidualEquation& interior,
      Dofs                             dofs)
    : interior_(interior),
      dofs_(std::move(dofs))
  {
  }

  Index numStates() const override
  {
    return interior_.numStates();
  }

  Index numParams() const override
  {
    return interior_.numParams();
  }

  Index numRes() const override
  {
    return interior_.numRes();
  }

  void res(const Vector& state,
           const Vector& params,
           Vector&       out) const override
  {
    interior_.res(state, params, out);
    for (Index dof : dofs_)
    {
      out[dof] = state[dof];
    }
  }

  void assembleStateJac(const Vector& state,
                        const Vector& params,
                        SystemMatrix& out) const override
  {
    interior_.assembleStateJac(state, params, out);
    for (Index row : dofs_)
    {
      for (Index col = 0; col < out.numCols(); ++col)
      {
        out.set(row, col, 0.0);
      }
      out.set(row, row, 1.0);
    }
  }

  void assembleParamJac(const Vector& state,
                        const Vector& params,
                        SystemMatrix& out) const override
  {
    interior_.assembleParamJac(state, params, out);
    for (Index row : dofs_)
    {
      for (Index col = 0; col < out.numCols(); ++col)
      {
        out.set(row, col, 0.0);
      }
    }
  }

private:
  const AssembledResidualEquation& interior_;
  Dofs                             dofs_;
};

void boundaryFluxResidual(Index       facet,
                          Index       num_qp,
                          Index       num_nodes,
                          Index       dim,
                          Index       num_res,
                          Index       num_states,
                          Index       num_params,
                          const Real* N,
                          const Real* point,
                          const Real* normal,
                          const Real* JxW,
                          const Real* state,
                          const Real* params,
                          Real*       out)
{
  (void) facet;
  (void) dim;
  (void) num_res;
  (void) num_states;
  (void) num_params;
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
      mq += Nq[b] * params[b];
    }

    for (Index a = 0; a < num_nodes; ++a)
    {
      out[a] -= Nq[a] * mq * JxW[iq];
    }
  }
}

class StateTrackingObjective final
  : public ObjectiveFunctional
{
public:
  StateTrackingObjective(const FESpace&         space,
                         Index                  num_params,
                         const Vector&          data,
                         const GaussQuadrature& quad,
                         Real                   weight)
    : space_(space),
      num_params_(num_params),
      data_(data),
      quad_(quad),
      weight_(weight)
  {
    if (data_.size() != space_.numDofs() || weight_ < 0.0)
    {
      throw std::runtime_error(
          "StateTrackingObjective received inconsistent data or weight");
    }
  }

  Index numStates() const override
  {
    return space_.numDofs();
  }

  Index numParams() const override
  {
    return num_params_;
  }

  Real value(const Vector& state,
             const Vector& params) const override
  {
    checkSizes(state, params);

    Real value_out = 0.0;
    integrateStateDifference(
        state,
        [](Index, Real, Real) {},
        value_out);
    return value_out;
  }

  void stateGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    checkSizes(state, params);
    resize(out, numStates());

    Real ignored = 0.0;
    integrateStateDifference(
        state,
        [&out](Index dof, Real value, Real)
        {
          out[dof] += value;
        },
        ignored);
  }

  void paramGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    checkSizes(state, params);
    resize(out, numParams());
  }

private:
  template <typename AddGradient>
  void integrateStateDifference(const Vector& state,
                                AddGradient   add_gradient,
                                Real&         value_out) const
  {
    ElementValues values(space_.finiteElement(), quad_);
    Dofs          dofs;

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

        for (Index a = 0; a < values.numDofs(); ++a)
        {
          const Index dof = dofs[a];
          add_gradient(dof, N[a] * diff_q * w, value_out);
        }
      }
    }
  }

  void checkSizes(const Vector& state,
                  const Vector& params) const
  {
    if (state.size() != numStates() || params.size() != numParams())
    {
      throw std::runtime_error("StateTrackingObjective size mismatch");
    }
  }

private:
  const FESpace&  space_;
  Index           num_params_{0};
  Vector          data_;
  GaussQuadrature quad_;
  Real            weight_{1.0};
};

class BoundaryMassRegularization final
  : public ObjectiveFunctional
{
public:
  BoundaryMassRegularization(
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
          "BoundaryMassRegularization received negative weight");
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

  Real value(const Vector& state,
             const Vector& params) const override
  {
    checkSizes(state, params);

    Real value_out = 0.0;
    integrate(params, [](Index, Real) {}, value_out);
    return value_out;
  }

  void stateGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    checkSizes(state, params);
    resize(out, numStates());
  }

  void paramGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    checkSizes(state, params);
    resize(out, numParams());

    Real ignored = 0.0;
    integrate(params, [&out](Index dof, Real value)
              { out[dof] += value; },
              ignored);
  }

private:
  template <typename AddGradient>
  void integrate(const Vector& params,
                 AddGradient   add_gradient,
                 Real&         value_out) const
  {
    BoundaryElementValues values(quad_);
    Dofs                  dofs;

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
          mq += N[a] * params[dofs[a]];
        }

        const Real w  = weight_ * values.JxW(iq);
        value_out    += 0.5 * mq * mq * w;

        for (Index a = 0; a < values.numNodes(); ++a)
        {
          add_gradient(dofs[a], N[a] * mq * w);
        }
      }
    }
  }

  void checkSizes(const Vector& state,
                  const Vector& params) const
  {
    if (state.size() != numStates() || params.size() != numParams())
    {
      throw std::runtime_error("BoundaryMassRegularization size mismatch");
    }
  }

private:
  const Mesh&       mesh_;
  BoundaryDofLayout param_layout_;
  Index             num_states_{0};
  GaussQuadrature   quad_;
  Real              weight_{0.0};
};

Vector boundaryCoordinates(const Mesh&              mesh,
                           const BoundaryDofLayout& layout)
{
  Vector x(layout.numDofs());
  Vector seen(layout.numDofs());
  Dofs   dofs;

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
        const Index node = facet.node_ids[a];
        x[dof]           = mesh.node(node)[0];
        seen[dof]        = 1.0;
      }
    }
  }
  return x;
}

Vector makeTrueFlux(const Vector& x)
{
  Vector flux(x.size());
  for (Index i = 0; i < flux.size(); ++i)
  {
    flux[i] = std::sin(2.0 * kPi * x[i]);
  }
  return flux;
}

Real rmse(const Vector& x, const Vector& y)
{
  if (x.size() != y.size())
  {
    throw std::runtime_error("rmse received incompatible vectors");
  }

  Real sum = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    const Real diff  = x[i] - y[i];
    sum             += diff * diff;
  }
  return std::sqrt(sum / x.size());
}

Vector difference(const Vector& x, const Vector& y)
{
  if (x.size() != y.size())
  {
    throw std::runtime_error("difference received incompatible vectors");
  }

  Vector diff(x.size());
  for (Index i = 0; i < x.size(); ++i)
  {
    diff[i] = x[i] - y[i];
  }
  return diff;
}

Vector boundaryFieldOnMesh(const Mesh&              mesh,
                           const BoundaryDofLayout& layout,
                           const Vector&            values)
{
  if (values.size() != layout.numDofs())
  {
    throw std::runtime_error("boundaryFieldOnMesh size mismatch");
  }

  Vector field(mesh.numNodes());
  Dofs   dofs;
  for (Index ib = 0; ib < layout.numFacets(); ++ib)
  {
    layout.facetDofs(ib, dofs);
    const auto& facet = layout.facet(ib);
    Index       a     = 0;
    for (Index node : facet.node_ids)
    {
      field[node] = values[dofs[a]];
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
                            const Vector&            observed_state,
                            const Vector&            true_params,
                            VisualizationOptions     options)
    : mesh_(mesh),
      param_layout_(param_layout),
      state_solver_(state_solver),
      observed_state_(observed_state),
      true_params_(true_params),
      true_param_field_(boundaryFieldOnMesh(mesh, param_layout, true_params)),
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

  void record(Index iter, const Vector& params, bool force = false)
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

    Vector state;
    state_solver_.solve(params, state);

    out_.beginStep(static_cast<Real>(iter));
    out_.addNodalScalarField("u", state);
    out_.addNodalScalarField("u_obs", observed_state_);
    out_.addNodalScalarField("u_error", difference(state, observed_state_));
    out_.addNodalScalarField("m", boundaryFieldOnMesh(mesh_, param_layout_, params));
    out_.addNodalScalarField("m_true", true_param_field_);
    out_.addNodalScalarField("m_error",
                             boundaryFieldOnMesh(mesh_,
                                                 param_layout_,
                                                 difference(params, true_params_)));

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
  const Mesh&              mesh_;
  const BoundaryDofLayout& param_layout_;
  StateSolver&             state_solver_;
  const Vector&            observed_state_;
  const Vector&            true_params_;
  Vector                   true_param_field_;
  VisualizationOptions     options_;
  TimeSeriesDataOut        out_;
  bool                     recorded_{false};
  Index                    last_iter_{-1};
};

Real DirectionalDerivative(
    ReducedFunctional& functional,
    const Vector&      params,
    const Vector&      dir,
    Real               eps)
{
  Vector plus(params.size());
  Vector minus(params.size());
  for (Index i = 0; i < params.size(); ++i)
  {
    plus[i]  = params[i] + eps * dir[i];
    minus[i] = params[i] - eps * dir[i];
  }
  return (functional.value(plus) - functional.value(minus)) / (2.0 * eps);
}

int run()
{
  const Index nx   = 8;
  const Index ny   = 8;
  const Real  beta = 1.0e-6;

  Mesh mesh = makeUnitSquareMesh(nx, ny);

  LagrangeQuadQ1 element;
  FESpace        space(&mesh, &element);
  space.setup();

  const auto volume_quad   = GaussQuadrature::quadrilateral(2);
  const auto boundary_quad = GaussQuadrature::segment(2);

  BoundaryDofLayout top_state_layout(space, kTopTag);
  BoundaryDofLayout top_param_layout =
      BoundaryDofLayout::compact(space, kTopTag);

  PoissonInteriorEquation interior(
      space, top_param_layout.numDofs(), volume_quad);

  EnzymeBoundaryIntegralKernel<boundaryFluxResidual>
      flux_kernel(mesh, boundary_quad, 2, 2, 2);

  BoundaryResidualEquation flux_eq(
      interior,
      top_state_layout,
      top_state_layout,
      top_param_layout,
      flux_kernel);

  DirichletResidualEquation equation(
      flux_eq, makeDirichletDofs(space));

  DenseSystemMatrix          state_jac;
  DenseLinearSolver          state_lin_solver;
  AssembledLinearStateSolver state_solver(
      equation, state_jac, state_lin_solver);

  const auto top_x  = boundaryCoordinates(mesh, top_param_layout);
  Vector     m_true = makeTrueFlux(top_x);

  Vector u_obs;
  state_solver.solve(m_true, u_obs);

  StateTrackingObjective tracking(
      space, top_param_layout.numDofs(), u_obs, volume_quad, 1.0);
  BoundaryMassRegularization regularization(
      mesh,
      top_param_layout,
      space.numDofs(),
      boundary_quad,
      beta);

  SumObjectiveFunctional objective(
      space.numDofs(), top_param_layout.numDofs());
  objective.add(tracking).add(regularization);

  DenseSystemMatrix           adj_jac;
  DenseLinearSolver           adj_lin_solver;
  MatrixEquationAdjointSolver adj_solver(
      equation, adj_jac, adj_lin_solver);

  AdjointReducedFunctional reduced(
      state_solver, adj_solver, equation, objective);

  Vector initial(top_param_layout.numDofs());
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

  Vector     initial_grad;
  const Real initial_value = reduced.valueGrad(initial, initial_grad);

  Vector dir(top_param_layout.numDofs());
  for (Index i = 0; i < dir.size(); ++i)
  {
    dir[i] = std::cos(0.73 * (i + 1));
  }
  const Real dir_norm = norm2(dir);
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
      [&history](const TaoIterationInfo& info, const Vector& params)
      {
        history.record(info.its, params);
      });

  TaoResult            result;
  const PetscErrorCode ierr = optimizer.solve(initial, result);
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        "TAO solve failed with PETSc error code "
        + std::to_string(ierr));
  }

  history.record(result.its, result.params, true);
  history.write();

  const Real final_rmse = rmse(result.params, m_true);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "2D Poisson boundary flux inverse problem\n";
  std::cout << "  mesh: " << nx << " x " << ny << " Q1 cells\n";
  std::cout << "  states: " << space.numDofs()
            << ", top flux parameters: " << top_param_layout.numDofs()
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
            << ", flux RMSE = " << final_rmse << '\n';
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

  std::cout << "\n  x            m_true       m_opt\n";
  for (Index i = 0; i < result.params.size(); ++i)
  {
    std::cout << "  " << std::setw(10)
              << top_x[i] << "  "
              << std::setw(10) << m_true[i] << "  "
              << std::setw(10) << result.params[i] << '\n';
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
    std::cerr << "poisson-boundary-flux-tao failed: " << e.what() << '\n';
    exit_code = 1;
  }

  ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS && exit_code == 0)
  {
    return 1;
  }
  return exit_code;
}
