#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "RunSupport.hpp"
#include <femx/common/Math.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/mesh/Mesh.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

namespace
{

void addBoundaryFacet(Mesh&                     mesh,
                      Index                     tag,
                      const std::string&        name,
                      const std::vector<Index>& node_ids)
{
  Mesh::BoundaryFacet facet;
  facet.dim           = 1;
  facet.entity_tag    = tag;
  facet.physical_tag  = tag;
  facet.physical_name = name;
  facet.shape         = Cell::Shape::Segment;
  facet.node_ids      = node_ids;
  mesh.addBoundaryFacet(std::move(facet));
}

navier_var::Params makeParams()
{
  navier_var::Params prm;

  navier_var::BCsParams inlet;
  inlet.name     = "inlet";
  inlet.physical = 7;
  inlet.ux       = 0.0;
  inlet.uy       = 0.0;

  navier_var::BCsParams outlet;
  outlet.name     = "outlet";
  outlet.physical = 8;
  outlet.p        = 0.0;

  navier_var::BCsParams wall;
  wall.name     = "wall";
  wall.physical = 9;
  wall.ux       = 0.0;
  wall.uy       = 0.0;

  prm.forward.bcs                  = {inlet, outlet, wall};
  prm.inverse.control.name         = "inlet";
  prm.inverse.bounds.axial_min     = 0.0;
  prm.inverse.bounds.axial_max     = 3.0;
  prm.inverse.bounds.fix_non_axial = true;
  prm.inverse.bounds.normal        = {1.0, 0.0, 0.0};
  return prm;
}

navier_var::TargetParams makeTarget()
{
  navier_var::TargetParams target;
  target.quantity        = "max_velocity";
  target.bulk_speed      = 2.0;
  target.pulse_amplitude = 0.0;
  target.period          = 1.0;
  target.radius          = 2.0;
  target.center          = {0.0, 1.0, 0.0};
  target.normal          = {1.0, 0.0, 0.0};
  return target;
}

} // namespace

class NavierVarRunSupportTests : public TestBase
{
public:
  TestOutcome controlDofsAndFixedBoundaries()
  {
    TestStatus status;
    status = true;

    Mesh mesh = Mesh::makeStructuredQuad(1, 1);
    addBoundaryFacet(mesh, 7, "inlet", {0, 2});
    addBoundaryFacet(mesh, 8, "outlet", {1, 3});
    addBoundaryFacet(mesh, 9, "wall", {0, 1});

    auto         elem  = navier_var::makeElement(mesh);
    MixedFESpace space = navier_var::makeSpace(mesh, *elem);
    const auto   u_dof = space.field(0);

    const navier_var::Params   prm = makeParams();
    const DirichletControl control =
        navier_var::makeVelocityControl(space, prm.inverse.control);

    status *= (control.numDofs() == 4);
    status *= (control.stateDof(0) == u_dof.globalDof(0, 0));
    status *= (control.stateDof(1) == u_dof.globalDof(0, 1));
    status *= (control.stateDof(2) == u_dof.globalDof(2, 0));
    status *= (control.stateDof(3) == u_dof.globalDof(2, 1));

    const Vector<Index> fixed = navier_var::fixedDofs(space, prm, control);

    status *= contains(fixed, space.field(1).globalDof(1, 0));
    status *= contains(fixed, u_dof.globalDof(1, 0));
    status *= !contains(fixed, u_dof.globalDof(0, 0));

    const Point3 center =
        navier_var::selectorCenter(mesh, prm.inverse.control);
    status *= isEqual(center[0], 0.0);
    status *= isEqual(center[1], 0.5);

    return status.report(__func__);
  }

  TestOutcome targetParamsBoundsAndMetrics()
  {
    TestStatus status;
    status = true;

    Mesh mesh = Mesh::makeStructuredQuad(1, 1);
    addBoundaryFacet(mesh, 7, "inlet", {0, 2});

    auto         elem  = navier_var::makeElement(mesh);
    MixedFESpace space = navier_var::makeSpace(mesh, *elem);

    const navier_var::Params   prm    = makeParams();
    const auto                 target = makeTarget();
    const DirichletControl control =
        navier_var::makeVelocityControl(space, prm.inverse.control);

    const Vector<Real> initial_prm =
        navier_var::initialControlParams(
            space, control, prm.forward.bcs[0], 1, 0.25);

    status *= (initial_prm.size() == control.numParams(1));
    status *= isEqual(initial_prm[0], 0.0);
    status *= isEqual(initial_prm[1], 0.0);
    status *= isEqual(initial_prm[2], 0.0);
    status *= isEqual(initial_prm[3], 0.0);

    const Vector<Real> true_prm =
        navier_var::makeTrueParams(space, control, target, 1, 0.25);

    status *= (true_prm.size() == control.numParams(1));
    status *= isEqual(true_prm[0], 1.5);
    status *= isEqual(true_prm[1], 0.0);
    status *= isEqual(true_prm[2], 2.0);
    status *= isEqual(true_prm[3], 0.0);

    Vector<Real> lower;
    Vector<Real> upper;
    navier_var::controlBounds(space,
                              control,
                              prm.inverse.bounds,
                              1,
                              lower,
                              upper);

    status *= isEqual(lower[0], 0.0);
    status *= isEqual(upper[0], 3.0);
    status *= isEqual(lower[1], 0.0);
    status *= isEqual(upper[1], 0.0);
    status *= isEqual(lower[2], 0.0);
    status *= isEqual(upper[2], 3.0);

    Vector<Real> zero(true_prm.size());
    zero.setZero();
    status *= isEqual(navier_var::blockRmse(control, true_prm, zero, 0),
                      1.25);
    status *= (navier_var::centerControlIndex(space, control, target) == 2);

    const Vector<Real> weights =
        navier_var::misfitW(2, 3.0);
    const Vector<Real> initial_weights =
        navier_var::misfitW(2, 3.0, true);
    status *= isEqual(weights[0], 0.0);
    status *= isEqual(weights[1], 3.0);
    status *= isEqual(initial_weights[0], 3.0);

    return status.report(__func__);
  }

  TestOutcome initialVelocityParameterLayoutAndBounds()
  {
    TestStatus status;
    status = true;

    Mesh mesh = Mesh::makeStructuredQuad(1, 1);
    addBoundaryFacet(mesh, 7, "inlet", {0, 2});

    auto         elem  = navier_var::makeElement(mesh);
    MixedFESpace space = navier_var::makeSpace(mesh, *elem);
    const auto   u_dof = space.field(0);
    const auto   p_dof = space.field(1);

    navier_var::Params prm = makeParams();
    prm.inverse.initial_velocity.enabled = true;
    prm.inverse.initial_velocity.lower   = -2.0;
    prm.inverse.initial_velocity.upper   = 2.0;

    const DirichletControl control =
        navier_var::makeVelocityControl(space, prm.inverse.control);
    const Vector<Index> velocity_dofs =
        navier_var::initialVelocityDofs(space);
    const navier_var::InverseParameterLayout layout =
        navier_var::inverseParameterLayout(
            space, control, prm.inverse.initial_velocity, 1);

    status *= (velocity_dofs.size() == mesh.numNodes() * mesh.dim());
    status *= (layout.initial_velocity_size == velocity_dofs.size());
    status *= (layout.control_offset == layout.initial_velocity_size);
    status *= (layout.control_size == control.numParams(1));
    status *= (layout.total_size
               == layout.initial_velocity_size + layout.control_size);

    Vector<Real> full_prm(layout.total_size);
    full_prm.setZero();
    for (Index i = 0; i < layout.initial_velocity_size; ++i)
    {
      full_prm[layout.initial_velocity_offset + i] =
          0.25 + 0.10 * static_cast<Real>(i);
    }

    Vector<Real> init_state;
    navier_var::initialStateFromParams(
        velocity_dofs, layout, space.numDofs(), full_prm, init_state);
    for (Index i = 0; i < layout.initial_velocity_size; ++i)
    {
      status *= isEqual(init_state[velocity_dofs[i]],
                        full_prm[layout.initial_velocity_offset + i]);
    }
    for (Index node = 0; node < mesh.numNodes(); ++node)
    {
      status *= isEqual(init_state[p_dof.globalDof(node, 0)], 0.0);
    }

    Vector<Real> state_grad(space.numDofs());
    for (Index i = 0; i < state_grad.size(); ++i)
    {
      state_grad[i] = -0.5 + 0.05 * static_cast<Real>(i);
    }
    Vector<Real> param_grad;
    navier_var::applyInitialVelocityParamJacT(
        velocity_dofs, layout, state_grad, param_grad);
    for (Index i = 0; i < layout.initial_velocity_size; ++i)
    {
      status *= isEqual(param_grad[layout.initial_velocity_offset + i],
                        state_grad[velocity_dofs[i]]);
    }
    for (Index i = 0; i < layout.control_size; ++i)
    {
      status *= isEqual(param_grad[layout.control_offset + i], 0.0);
    }

    Vector<Real> init_prm =
        navier_var::initialInverseParams(
            space, control, prm, layout, 1, 0.25);
    status *= (init_prm.size() == layout.total_size);
    for (Index i = 0; i < layout.initial_velocity_size; ++i)
    {
      status *= isEqual(init_prm[layout.initial_velocity_offset + i],
                        0.0);
    }

    Vector<Real> lower;
    Vector<Real> upper;
    navier_var::inverseBounds(
        space, control, prm, layout, 1, lower, upper);
    status *= isEqual(lower[layout.initial_velocity_offset], -2.0);
    status *= isEqual(upper[layout.initial_velocity_offset], 2.0);
    status *= isEqual(lower[layout.control_offset + 0], 0.0);
    status *= isEqual(upper[layout.control_offset + 0], 3.0);
    status *= isEqual(lower[layout.control_offset + 1], 0.0);
    status *= isEqual(upper[layout.control_offset + 1], 0.0);
    status *= (u_dof.globalDof(0, 0) == velocity_dofs[0]);

    return status.report(__func__);
  }

  TestOutcome gridObservationOperatorSamplesVelocity()
  {
    TestStatus status;
    status = true;

    Mesh mesh = Mesh::makeStructuredQuad(1, 1);
    auto         elem  = navier_var::makeElement(mesh);
    MixedFESpace space = navier_var::makeSpace(mesh, *elem);

    navier_var::ObservationParams obs_prm;
    obs_prm.type = "grid";
    obs_prm.components = {0};
    obs_prm.grid = navier_var::ObservationParams::Grid{};
    obs_prm.grid->lower  = {0.0, 0.0, 0.0};
    obs_prm.grid->upper  = {1.0, 1.0, 0.0};
    obs_prm.grid->counts = {2, 2, 1};

    const auto obs =
        navier_var::makeObs(space, obs_prm, 1, space.numDofs(), 0);
    inverse::TimeObservationData obs_data(2, obs->numObservations());
    navier_var::setObsLayout(obs_data, space, obs_prm);
    const auto file_obs =
        navier_var::makeObsFromData(space, obs_data, 1, space.numDofs(), 0);

    status *= obs->numObservations() == 4;
    status *= file_obs->numObservations() == 4;

    Vector<Real> state(space.numDofs());
    const auto   u_dof = space.field(0);
    for (Index node = 0; node < mesh.numNodes(); ++node)
    {
      const Point3 point = mesh.node(node);
      state[u_dof.globalDof(node, 0)] = point[0] + 2.0 * point[1];
      state[u_dof.globalDof(node, 1)] = -1.0;
    }

    Vector<Real> prm;
    Vector<Real> values;
    file_obs->observe(0, state, prm, values);

    status *= isEqual(values[0], 0.0);
    status *= isEqual(values[1], 1.0);
    status *= isEqual(values[2], 2.0);
    status *= isEqual(values[3], 3.0);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running ns-var run support tests:\n";

  femx::tests::NavierVarRunSupportTests test;

  femx::tests::TestingResults result;
  result += test.controlDofsAndFixedBoundaries();
  result += test.targetParamsBoundsAndMetrics();
  result += test.initialVelocityParameterLayoutAndBounds();
  result += test.gridObservationOperatorSamplesVelocity();

  return result.summary();
}
