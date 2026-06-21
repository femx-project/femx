#include "Helper.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/common/Math.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/fem/ObservationGrid.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/fem/VelocityProfile.hpp>

using namespace femx::fem;
using namespace femx::problem;
using namespace femx::solve;

namespace femx::navier_var_new
{

namespace
{

constexpr Index kQuadOrder = 2;

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

Point3 toPoint(const std::array<Real, 3>& values)
{
  return {values[0], values[1], values[2]};
}

Index observationSolveSteps(const Params& prm)
{
  // Level 0 is the controlled initial condition before the first observation.
  // Observation levels are mapped to solve levels 1..steps.
  return prm.fwd.time.steps;
}

GaussQuadrature makeVelocityQuadrature(const MixedFESpace& space)
{
  return GaussQuadrature::make(
      space.field(0).space().finiteElement().referenceElement(),
      kQuadOrder);
}

NavierKernel makeAppKernel(const MixedFESpace&    space,
                           const GaussQuadrature& quad,
                           const FluidParams&     fluid,
                           Real                   dt)
{
  return makeNavierKernel(space.field(0).space(),
                          quad,
                          space.numDofsPerElem(),
                          fluid.rho,
                          fluid.mu,
                          dt);
}

Vector<Index> obsComponents(const MixedFESpace&      space,
                            const ObservationParams& prm)
{
  Vector<Index> components;
  const auto    field = space.field(0);
  if (prm.components.empty())
  {
    components.reserve(field.numComponents());
    for (Index component = 0; component < field.numComponents(); ++component)
    {
      components.push_back(component);
    }
    return components;
  }

  components.reserve(static_cast<Index>(prm.components.size()));
  for (Index component : prm.components)
  {
    if (component < 0 || component >= field.numComponents())
    {
      throw std::runtime_error("observation component is out of range");
    }
    components.push_back(component);
  }
  return components;
}

std::vector<Point3> gridObsPoints(const ObservationParams& prm)
{
  if (!prm.grid)
  {
    throw std::runtime_error("grid observation requires inverse.obs.grid");
  }

  const auto& grid = *prm.grid;
  if (grid.use_spacing)
  {
    return observationGridPoints(
        toPoint(grid.origin), grid.counts, toPoint(grid.spacing));
  }
  return observationGridPoints(
      toPoint(grid.lower), toPoint(grid.upper), grid.counts);
}

std::vector<Point3> obsPoints(const ObservationParams& prm)
{
  if (prm.type == "grid")
  {
    return gridObsPoints(prm);
  }

  throw std::runtime_error(
      "Unsupported observation type: " + prm.type
      + " (expected 'grid')");
}

std::vector<Point3> activeObsPoints(const MixedFESpace&      space,
                                    const ObservationParams& prm)
{
  if (prm.type != "grid")
  {
    return obsPoints(prm);
  }

  const std::vector<Point3> points = gridObsPoints(prm);
  std::vector<Point3>       filtered =
      TimePointInterpolator::filterPointsInside(space, 0, points);
  if (filtered.empty())
  {
    throw std::runtime_error(
        "observation grid mask removed all points outside the fluid mesh");
  }
  return filtered;
}

bool matchesBoundaryConfig(const BoundarySelector& selector,
                           const BCsParams&        bc)
{
  if (!selector.name.empty())
  {
    return bc.name == selector.name;
  }
  return bc.physical == selector.physical;
}

bool matchesBoundaryFacet(const BoundarySelector&    selector,
                          const Mesh::BoundaryFacet& facet)
{
  if (!selector.name.empty())
  {
    return facet.physical_name == selector.name;
  }
  return facet.physical_tag == selector.physical;
}

std::vector<ControlSpatialRegularization::Edge> controlSpatialEdges(
    const MixedFESpace&     space,
    const BoundarySelector& selector,
    const DirichletControl& control)
{
  const auto                               u_dof      = space.field(0);
  const Index                              components = u_dof.numComponents();
  std::map<std::pair<Index, Index>, Index> ctr_index;

  for (Index i = 0; i < control.numDofs(); ++i)
  {
    const Index dof       = control.stateDof(i);
    const Index node      = dof / components;
    const Index component = dof % components;
    if (node < 0
        || node >= space.mesh().numNodes()
        || u_dof.globalDof(node, component) != dof)
    {
      throw std::runtime_error(
          "controlSpatialEdges expected velocity control dofs");
    }
    ctr_index[{node, component}] = i;
  }

  std::set<ControlSpatialRegularization::Edge> unique_edges;
  const auto                                   add_edge =
      [&](Index node0, Index node1, Index component)
  {
    const auto it0 = ctr_index.find({node0, component});
    const auto it1 = ctr_index.find({node1, component});
    if (it0 == ctr_index.end() || it1 == ctr_index.end())
    {
      return;
    }

    Index a = it0->second;
    Index b = it1->second;
    if (a == b)
    {
      return;
    }
    if (b < a)
    {
      std::swap(a, b);
    }
    unique_edges.insert({a, b});
  };

  for (const auto& facet : space.mesh().boundaryFacets())
  {
    if (!matchesBoundaryFacet(selector, facet)
        || facet.node_ids.size() < 2)
    {
      continue;
    }

    for (Index component = 0; component < components; ++component)
    {
      for (std::size_t i = 1; i < facet.node_ids.size(); ++i)
      {
        add_edge(facet.node_ids[i - 1], facet.node_ids[i], component);
      }
      if (facet.node_ids.size() > 2)
      {
        add_edge(facet.node_ids.back(), facet.node_ids.front(), component);
      }
    }
  }

  return {unique_edges.begin(), unique_edges.end()};
}

bool hasFixedVelocityValue(const BCsParams& bc)
{
  return bc.ux || bc.uy || bc.uz || bc.vel;
}

std::optional<Real> constantVelocityComponent(
    const BCsParams& bc,
    Index            component)
{
  if (component == 0 && bc.ux)
  {
    return *bc.ux;
  }
  if (component == 1 && bc.uy)
  {
    return *bc.uy;
  }
  if (component == 2 && bc.uz)
  {
    return *bc.uz;
  }
  return std::nullopt;
}

const BCsParams& pressureGaugeBoundary(const Params& prm)
{
  for (const auto& bc : prm.fwd.bcs)
  {
    if (bc.p)
    {
      return bc;
    }
  }
  throw std::runtime_error("simulation.bcs must contain a pressure boundary");
}

void addFixedValue(std::map<Index, Vector<Real>>& values,
                   Index                          dof,
                   Index                          step,
                   Index                          steps,
                   Real                           value)
{
  auto it = values.find(dof);
  if (it == values.end())
  {
    it = values.emplace(dof, Vector<Real>(steps)).first;
    for (Index i = 0; i < steps; ++i)
    {
      it->second[i] = std::numeric_limits<Real>::quiet_NaN();
    }
  }
  else if (!std::isnan(it->second[step])
           && std::abs(it->second[step] - value) > 1.0e-12)
  {
    throw std::runtime_error(
        "fixed boundary has conflicting values at dof "
        + std::to_string(dof) + ", step " + std::to_string(step));
  }
  it->second[step] = value;
}

void addPressureGaugeValues(const MixedFESpace&            space,
                            const BCsParams&               bc,
                            Index                          steps,
                            std::map<Index, Vector<Real>>& values)
{
  if (!bc.p)
  {
    return;
  }

  const Vector<Index> dofs = gaugeDofs(space, bcSelector(bc));
  for (Index step = 0; step < steps; ++step)
  {
    addFixedValue(values, dofs[0], step, steps, *bc.p);
  }
}

Real profileVelocityValue(const MixedFESpace&     space,
                          const DirichletControl& control,
                          const TargetParams&     target,
                          Real                    time,
                          Index                   i);

void addProfileVelocityValues(const MixedFESpace&            space,
                              const DirichletControl&        control,
                              const BCsParams&               bc,
                              Index                          steps,
                              Real                           dt,
                              std::map<Index, Vector<Real>>& values)
{
  if (!bc.vel)
  {
    return;
  }

  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < control.numDofs(); ++i)
    {
      addFixedValue(values,
                    control.stateDof(i),
                    step,
                    steps,
                    profileVelocityValue(
                        space,
                        control,
                        *bc.vel,
                        static_cast<Real>(step + 1) * dt,
                        i));
    }
  }
}

Vector<Index> fixedvdofsForBc(const MixedFESpace& space,
                              const BCsParams&    bc)
{
  const DirichletControl boundary =
      makeVelocityControl(space, bcSelector(bc));
  if (bc.vel)
  {
    return boundary.stateDofs();
  }

  const auto    u_dof = space.field(0);
  const Index   nd    = u_dof.numComponents();
  Vector<Index> dofs;
  for (Index dof : boundary.stateDofs())
  {
    const Index component = dof % nd;
    if (constantVelocityComponent(bc, component))
    {
      dofs.push_back(dof);
    }
  }
  return dofs;
}

DirichletControl makeBoundaryControl(const MixedFESpace& space,
                                     const Params&       prm,
                                     const BCsParams&    bc)
{
  const DirichletControl boundary =
      makeVelocityControl(space, bcSelector(bc));

  Vector<Index> fixed_velocity_dofs;
  for (const auto& fixed_bc : prm.fwd.bcs)
  {
    if (matchesBoundaryConfig(prm.inv.ctr, fixed_bc)
        || !hasFixedVelocityValue(fixed_bc))
    {
      continue;
    }
    appendUniqueExcept(
        fixed_velocity_dofs, fixedvdofsForBc(space, fixed_bc), {});
  }

  Vector<Index> active_dofs;
  active_dofs.reserve(boundary.numDofs());
  for (Index dof : boundary.stateDofs())
  {
    if (!contains(fixed_velocity_dofs, dof))
    {
      active_dofs.push_back(dof);
    }
  }
  if (active_dofs.empty())
  {
    throw std::runtime_error(
        "control boundary has no active velocity dofs after fixed-boundary exclusion");
  }
  return DirichletControl(std::move(active_dofs));
}

void addConstantVelocityValues(const MixedFESpace&            space,
                               const DirichletControl&        control,
                               const BCsParams&               bc,
                               Index                          steps,
                               std::map<Index, Vector<Real>>& values)
{
  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  for (Index i = 0; i < control.numDofs(); ++i)
  {
    const Index dof       = control.stateDof(i);
    const Index component = dof % nd;
    const auto  value     = constantVelocityComponent(bc, component);
    if (!value)
    {
      continue;
    }
    for (Index step = 0; step < steps; ++step)
    {
      addFixedValue(values, dof, step, steps, *value);
    }
  }
}

Real profileVelocityValue(const MixedFESpace&     space,
                          const DirichletControl& control,
                          const TargetParams&     target,
                          Real                    time,
                          Index                   i)
{
  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  const Index dof   = control.stateDof(i);
  const Index node  = dof / nd;
  const Index comp  = dof - nd * node;

  const auto profile = poiseuilleProfile(
      target.center, target.normal, target.radius);
  if (comp >= static_cast<Index>(profile.normal.size()))
  {
    return 0.0;
  }

  const Real pulse =
      sinePulseFactor(time, target.pulse_amplitude, target.period);
  const Real peak_speed =
      peakSpeed(
          target.quantity, "poiseuille", target.bulk_speed, 1.0, 1.5)
      * pulse;
  return velocityComponent(
      profile, space.mesh().node(node), peak_speed, comp);
}

Real controlStepTime(Index step,
                     Real  dt)
{
  return static_cast<Real>(step + 1) * dt;
}

Vector<Real> rawObservationTimes(const TimeObservationData& data)
{
  if (data.numLevels() <= 0)
  {
    throw std::runtime_error("observation data has no time levels");
  }

  Vector<Real> times(data.numLevels());
  for (Index row = 0; row < data.numLevels(); ++row)
  {
    times[row] =
        data.hasTimeValues()
            ? data.timeValue(row)
            : static_cast<Real>(data.timeLevel(row));
    if (!std::isfinite(times[row])
        || (row > 0 && times[row] <= times[row - 1]))
    {
      throw std::runtime_error(
          "observation times must be finite and increasing");
    }
  }
  return times;
}

Vector<Real> observationTimesOnSolveLevels(const TimeObservationData& data,
                                           Real                       dt,
                                           Index                      steps)
{
  if (steps <= 0 || dt <= 0.0 || !std::isfinite(dt))
  {
    throw std::runtime_error(
        "observationTimesOnSolveLevels received invalid inputs");
  }

  const Vector<Real> raw_times = rawObservationTimes(data);
  Vector<Real>       times(raw_times.size());
  const Real         first_solve_time = dt;
  const Real         final_solve_time = static_cast<Real>(steps) * dt;
  if (raw_times.size() == 1)
  {
    times[0] = first_solve_time;
    return times;
  }
  if (steps <= 1)
  {
    throw std::runtime_error(
        "multiple observation times require at least two solve steps");
  }

  const Real raw_span = raw_times.back() - raw_times.front();
  const Real sim_span = final_solve_time - first_solve_time;
  if (raw_span <= 0.0 || sim_span <= 0.0)
  {
    throw std::runtime_error("observation time span is invalid");
  }

  for (Index row = 0; row < raw_times.size(); ++row)
  {
    const Real s = (raw_times[row] - raw_times.front()) / raw_span;
    times[row]   = first_solve_time + s * sim_span;
    if (!std::isfinite(times[row])
        || (row > 0 && times[row] <= times[row - 1]))
    {
      throw std::runtime_error(
          "mapped observation times must be finite and increasing");
    }
  }
  return times;
}

TimeObservationData observationDataOnSolveLevels(
    const TimeObservationData& data,
    Real                       dt,
    Index                      steps)
{
  TimeObservationData out = data;
  out.setTimeValues(observationTimesOnSolveLevels(data, dt, steps));
  return out;
}

Vector<Real> controlKnotTimes(const TimeObservationData& data,
                              Real                       dt,
                              Index                      steps)
{
  return observationTimesOnSolveLevels(data, dt, steps);
}

Vector<LinearInterpolation> controlTimeStencils(
    Index               steps,
    Real                dt,
    const Vector<Real>& times)
{
  if (steps <= 0)
  {
    throw std::runtime_error("controlTimeStencils requires positive steps");
  }

  Vector<LinearInterpolation> stencils(steps);
  for (Index step = 0; step < steps; ++step)
  {
    stencils[step] = linearInterpolation(times, controlStepTime(step, dt));
  }
  return stencils;
}

void addInitialVelocityValue(std::map<Index, Real>& values,
                             Index                  dof,
                             Real                   value)
{
  const auto [it, inserted] = values.emplace(dof, value);
  if (!inserted && std::abs(it->second - value) > 1.0e-12)
  {
    throw std::runtime_error(
        "initial velocity boundary has conflicting values at dof "
        + std::to_string(dof));
  }
}

void addInitialProfileVelocityValues(const MixedFESpace&     space,
                                     const DirichletControl& control,
                                     const TargetParams&     target,
                                     Real                    time,
                                     std::map<Index, Real>&  values)
{
  for (Index i = 0; i < control.numDofs(); ++i)
  {
    addInitialVelocityValue(values,
                            control.stateDof(i),
                            profileVelocityValue(
                                space, control, target, time, i));
  }
}

void addInitialConstantVelocityValues(const MixedFESpace&     space,
                                      const DirichletControl& control,
                                      const BCsParams&        bc,
                                      bool                    fill_missing,
                                      std::map<Index, Real>&  values)
{
  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  for (Index i = 0; i < control.numDofs(); ++i)
  {
    const Index dof       = control.stateDof(i);
    const Index component = dof % nd;
    const auto  value     = constantVelocityComponent(bc, component);
    if (!value && !fill_missing)
    {
      continue;
    }
    addInitialVelocityValue(values, dof, value.value_or(0.0));
  }
}

void addInitialBoundaryVelocityValues(const MixedFESpace&     space,
                                      const DirichletControl& control,
                                      const BCsParams&        bc,
                                      bool                    fill_missing,
                                      std::map<Index, Real>&  values)
{
  if (bc.vel)
  {
    addInitialProfileVelocityValues(
        space, control, *bc.vel, 0.0, values);
    return;
  }
  addInitialConstantVelocityValues(
      space, control, bc, fill_missing, values);
}

FixedDofValues toFixedDofValues(
    const std::map<Index, Vector<Real>>& values,
    Index                                steps)
{
  FixedDofValues out;
  out.dofs.reserve(static_cast<Index>(values.size()));
  for (const auto& entry : values)
  {
    out.dofs.push_back(entry.first);
  }

  out.values.resize(steps * out.dofs.size());
  Index i = 0;
  for (const auto& entry : values)
  {
    for (Index step = 0; step < steps; ++step)
    {
      if (std::isnan(entry.second[step]))
      {
        throw std::runtime_error(
            "fixed boundary value was not assigned for every time step");
      }
      out.values[step * out.dofs.size() + i] = entry.second[step];
    }
    ++i;
  }
  return out;
}

} // namespace

void initialStateFromParams(
    const Vector<Index>&          velocity_dofs,
    const InverseParameterLayout& layout,
    const Vector<Real>&           base_state,
    const Vector<Real>&           prm,
    Vector<Real>&                 out);

InitialVelocityStateSolver::InitialVelocityStateSolver(
    TimeLinearStateSolver& solver,
    Vector<Index>          velocity_dofs,
    InverseParameterLayout layout,
    Vector<Real>           x0)
  : solver_(solver),
    velocity_dofs_(std::move(velocity_dofs)),
    layout_(layout),
    x0_(std::move(x0))
{
  if (!layout_.hasInitialVelocity())
  {
    throw std::runtime_error(
        "InitialVelocityStateSolver requires initial velocity parameters");
  }
  if (solver_.numParams() != layout_.total_size)
  {
    throw std::runtime_error(
        "InitialVelocityStateSolver parameter size mismatch");
  }
  if (velocity_dofs_.size() != layout_.init_vel_size)
  {
    throw std::runtime_error(
        "InitialVelocityStateSolver velocity dof size mismatch");
  }
  if (x0_.empty())
  {
    x0_.resize(solver_.numStates());
  }
  else if (x0_.size() != solver_.numStates())
  {
    throw std::runtime_error(
        "InitialVelocityStateSolver base initial state size mismatch");
  }
}

Index InitialVelocityStateSolver::numSteps() const
{
  return solver_.numSteps();
}

Index InitialVelocityStateSolver::numStates() const
{
  return solver_.numStates();
}

Index InitialVelocityStateSolver::numParams() const
{
  return solver_.numParams();
}

void InitialVelocityStateSolver::solve(const Vector<Real>& prm,
                                       TimeTrajectory&     tr)
{
  initialStateFromParams(
      velocity_dofs_, layout_, x0_, prm, initial_state_);
  solver_.setInitialState(initial_state_);
  solver_.solve(prm, tr);
}

ParameterSliceTimeObjective::ParameterSliceTimeObjective(
    const TimeObjective& base,
    Index                total_params,
    Index                offset)
  : base_(base),
    total_params_(total_params),
    offset_(offset)
{
  if (total_params_ < 0 || offset_ < 0
      || offset_ + base_.numParams() > total_params_)
  {
    throw std::runtime_error(
        "ParameterSliceTimeObjective received invalid parameter slice");
  }
}

Index ParameterSliceTimeObjective::numSteps() const
{
  return base_.numSteps();
}

Index ParameterSliceTimeObjective::numStates() const
{
  return base_.numStates();
}

Index ParameterSliceTimeObjective::numParams() const
{
  return total_params_;
}

Real ParameterSliceTimeObjective::value(const TimeTrajectory& tr,
                                        const Vector<Real>&   prm) const
{
  return base_.value(tr, slice(prm));
}

void ParameterSliceTimeObjective::stateGrad(Index                 level,
                                            const TimeTrajectory& tr,
                                            const Vector<Real>&   prm,
                                            Vector<Real>&         out) const
{
  base_.stateGrad(level, tr, slice(prm), out);
}

void ParameterSliceTimeObjective::paramGrad(const TimeTrajectory& tr,
                                            const Vector<Real>&   prm,
                                            Vector<Real>&         out) const
{
  const Vector<Real> sub_prm = slice(prm);
  Vector<Real>       sub_grad;
  base_.paramGrad(tr, sub_prm, sub_grad);
  if (sub_grad.size() != base_.numParams())
  {
    throw std::runtime_error(
        "ParameterSliceTimeObjective base parameter gradient size mismatch");
  }

  resize(out, total_params_);
  for (Index i = 0; i < sub_grad.size(); ++i)
  {
    out[offset_ + i] = sub_grad[i];
  }
}

Vector<Real> ParameterSliceTimeObjective::slice(
    const Vector<Real>& prm) const
{
  if (prm.size() != total_params_)
  {
    throw std::runtime_error(
        "ParameterSliceTimeObjective parameter size mismatch");
  }

  Vector<Real> out(base_.numParams());
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = prm[offset_ + i];
  }
  return out;
}

InitialVelocityRegularization::InitialVelocityRegularization(
    Index                  num_steps,
    Index                  num_states,
    InverseParameterLayout layout,
    Real                   beta)
  : num_steps_(num_steps),
    num_states_(num_states),
    layout_(layout),
    beta_(beta)
{
  if (num_steps_ < 0 || num_states_ < 0 || layout_.total_size < 0
      || layout_.init_vel_offset < 0
      || layout_.init_vel_size < 0
      || layout_.init_vel_offset + layout_.init_vel_size
             > layout_.total_size
      || beta_ < 0.0)
  {
    throw std::runtime_error(
        "InitialVelocityRegularization received invalid inputs");
  }
}

Index InitialVelocityRegularization::numSteps() const
{
  return num_steps_;
}

Index InitialVelocityRegularization::numStates() const
{
  return num_states_;
}

Index InitialVelocityRegularization::numParams() const
{
  return layout_.total_size;
}

Real InitialVelocityRegularization::value(const TimeTrajectory& tr,
                                          const Vector<Real>&   prm) const
{
  (void) tr;
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "InitialVelocityRegularization parameter size mismatch");
  }

  Real out = 0.0;
  for (Index i = 0; i < layout_.init_vel_size; ++i)
  {
    const Real value  = prm[layout_.init_vel_offset + i];
    out              += 0.5 * beta_ * value * value;
  }
  return out;
}

void InitialVelocityRegularization::stateGrad(
    Index                 level,
    const TimeTrajectory& tr,
    const Vector<Real>&   prm,
    Vector<Real>&         out) const
{
  (void) level;
  (void) tr;
  (void) prm;
  resize(out, num_states_);
}

void InitialVelocityRegularization::paramGrad(
    const TimeTrajectory& tr,
    const Vector<Real>&   prm,
    Vector<Real>&         out) const
{
  (void) tr;
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "InitialVelocityRegularization parameter size mismatch");
  }

  resize(out, numParams());
  for (Index i = 0; i < layout_.init_vel_size; ++i)
  {
    const Index idx = layout_.init_vel_offset + i;
    out[idx]        = beta_ * prm[idx];
  }
}

ControlSpatialRegularization::ControlSpatialRegularization(
    Index             num_steps,
    Index             num_states,
    Index             num_params,
    Index             ctr_offset,
    Index             ctr_levels,
    Index             ctr_dofs,
    std::vector<Edge> edges,
    Real              beta)
  : num_steps_(num_steps),
    num_states_(num_states),
    num_params_(num_params),
    ctr_offset_(ctr_offset),
    ctr_levels_(ctr_levels),
    ctr_dofs_(ctr_dofs),
    edges_(std::move(edges)),
    beta_(beta)
{
  if (num_steps_ < 0 || num_states_ < 0 || num_params_ < 0
      || ctr_offset_ < 0 || ctr_levels_ < 0
      || ctr_dofs_ < 0 || beta_ < 0.0
      || ctr_offset_ + ctr_levels_ * ctr_dofs_ > num_params_)
  {
    throw std::runtime_error(
        "ControlSpatialRegularization received invalid inputs");
  }
  if (beta_ > 0.0 && edges_.empty())
  {
    throw std::runtime_error(
        "ControlSpatialRegularization found no boundary edges");
  }
}

Index ControlSpatialRegularization::numSteps() const
{
  return num_steps_;
}

Index ControlSpatialRegularization::numStates() const
{
  return num_states_;
}

Index ControlSpatialRegularization::numParams() const
{
  return num_params_;
}

Real ControlSpatialRegularization::value(
    const TimeTrajectory& tr,
    const Vector<Real>&   prm) const
{
  (void) tr;
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "ControlSpatialRegularization parameter size mismatch");
  }

  Real out = 0.0;
  for (Index step = 0; step < ctr_levels_; ++step)
  {
    for (const Edge& edge : edges_)
    {
      const Real diff =
          prm[paramIndex(step, edge.first)]
          - prm[paramIndex(step, edge.second)];
      out += 0.5 * beta_ * diff * diff;
    }
  }
  return out;
}

void ControlSpatialRegularization::stateGrad(
    Index                 level,
    const TimeTrajectory& tr,
    const Vector<Real>&   prm,
    Vector<Real>&         out) const
{
  (void) level;
  (void) tr;
  (void) prm;
  resize(out, num_states_);
}

void ControlSpatialRegularization::paramGrad(
    const TimeTrajectory& tr,
    const Vector<Real>&   prm,
    Vector<Real>&         out) const
{
  (void) tr;
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "ControlSpatialRegularization parameter size mismatch");
  }

  resize(out, num_params_);
  for (Index step = 0; step < ctr_levels_; ++step)
  {
    for (const Edge& edge : edges_)
    {
      const Index a     = paramIndex(step, edge.first);
      const Index b     = paramIndex(step, edge.second);
      const Real  diff  = prm[a] - prm[b];
      out[a]           += beta_ * diff;
      out[b]           -= beta_ * diff;
    }
  }
}

Index ControlSpatialRegularization::paramIndex(
    Index step,
    Index ctr_index) const
{
  return ctr_offset_ + step * ctr_dofs_ + ctr_index;
}

Vector<Index> initialvdofs(const MixedFESpace& space)
{
  const auto  u_dof = space.field(0);
  const Index nodes = space.mesh().numNodes();
  const Index comps = u_dof.numComponents();

  Vector<Index> dofs;
  dofs.reserve(nodes * comps);
  for (Index node = 0; node < nodes; ++node)
  {
    for (Index comp = 0; comp < comps; ++comp)
    {
      dofs.push_back(u_dof.globalDof(node, comp));
    }
  }
  return dofs;
}

Vector<Index> initialvdofs(const MixedFESpace&     space,
                           const Params&           prm,
                           const DirichletControl& control)
{
  Vector<Index> constrained = control.stateDofs();
  for (const auto& bc : prm.fwd.bcs)
  {
    if (matchesBoundaryConfig(prm.inv.ctr, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }
    appendUniqueExcept(
        constrained, fixedvdofsForBc(space, bc), control.stateDofs());
  }

  const Vector<Index> all = initialvdofs(space);
  Vector<Index>       out;
  out.reserve(all.size());
  for (Index dof : all)
  {
    if (!contains(constrained, dof))
    {
      out.push_back(dof);
    }
  }
  return out;
}

Vector<Real> initialVelocityBoundaryState(
    const MixedFESpace&     space,
    const Params&           prm,
    const DirichletControl& control)
{
  std::map<Index, Real> values;

  addInitialBoundaryVelocityValues(
      space, control, controlBoundary(prm), true, values);

  for (const auto& bc : prm.fwd.bcs)
  {
    if (matchesBoundaryConfig(prm.inv.ctr, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }

    const Vector<Index> fixed_dofs = fixedvdofsForBc(space, bc);
    Vector<Index>       active_dofs;
    for (Index dof : fixed_dofs)
    {
      if (!contains(control.stateDofs(), dof))
      {
        active_dofs.push_back(dof);
      }
    }
    if (active_dofs.empty())
    {
      continue;
    }
    const DirichletControl active_fixed(active_dofs);
    addInitialBoundaryVelocityValues(
        space, active_fixed, bc, false, values);
  }

  Vector<Real> out(space.numDofs());
  out.setZero();
  for (const auto& entry : values)
  {
    if (entry.first < 0 || entry.first >= out.size())
    {
      throw std::runtime_error(
          "initial velocity boundary dof is out of range");
    }
    out[entry.first] = entry.second;
  }
  return out;
}

InverseParameterLayout inverseParameterLayout(
    const MixedFESpace&          space,
    const DirichletControl&      control,
    const InitialVelocityParams& initial_velocity,
    Index                        steps,
    Index                        init_vel_size,
    Index                        ctr_levels)
{
  (void) space;
  if (steps <= 0 || init_vel_size < 0 || ctr_levels <= 0)
  {
    throw std::runtime_error(
        "inverseParameterLayout received invalid sizes");
  }

  InverseParameterLayout layout;
  if (initial_velocity.enabled)
  {
    layout.init_vel_offset = 0;
    layout.init_vel_size   = init_vel_size;
  }
  layout.ctr_offset = layout.init_vel_offset
                      + layout.init_vel_size;
  layout.ctr_levels = ctr_levels;
  layout.ctr_size   = ctr_levels * control.numDofs();
  layout.total_size = layout.ctr_offset + layout.ctr_size;
  return layout;
}

FixedDofValues fixedDofValues(const MixedFESpace&     space,
                              const Params&           prm,
                              const DirichletControl& control,
                              Index                   steps,
                              Real                    dt)
{
  if (steps <= 0)
  {
    throw std::runtime_error("fixedDofValues requires positive steps");
  }

  std::map<Index, Vector<Real>> values;
  addPressureGaugeValues(
      space, pressureGaugeBoundary(prm), steps, values);

  for (const auto& bc : prm.fwd.bcs)
  {
    if (matchesBoundaryConfig(prm.inv.ctr, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }

    const Vector<Index> fixed_dofs = fixedvdofsForBc(space, bc);
    Vector<Index>       active_dofs;
    for (Index dof : fixed_dofs)
    {
      if (!contains(control.stateDofs(), dof))
      {
        active_dofs.push_back(dof);
      }
    }
    if (active_dofs.empty())
    {
      continue;
    }
    const DirichletControl active_fixed(active_dofs);

    addProfileVelocityValues(space,
                             active_fixed,
                             bc,
                             steps,
                             dt,
                             values);
    addConstantVelocityValues(
        space, active_fixed, bc, steps, values);
  }

  return toFixedDofValues(values, steps);
}

std::unique_ptr<TimeObservationOperator> makeObs(
    const MixedFESpace&      space,
    const ObservationParams& prm,
    Index                    steps,
    Index                    num_states,
    Index                    num_prm)
{
  if (num_states != space.numDofs())
  {
    throw std::runtime_error("makeObs state size does not match FEM space");
  }

  return std::make_unique<TimePointInterpolator>(
      steps,
      space,
      0,
      activeObsPoints(space, prm),
      obsComponents(space, prm),
      num_prm);
}

void setObsLayout(TimeObservationData&     data,
                  const MixedFESpace&      space,
                  const ObservationParams& prm)
{
  data.setLayout(
      "point", activeObsPoints(space, prm), obsComponents(space, prm));
}

Vector<Real> wError(Index num_steps,
                    Real  weight,
                    bool  include_initial)
{
  Vector<Real> weights(num_steps + 1);
  weights[0] = include_initial ? weight : 0.0;
  for (Index level = 1; level <= num_steps; ++level)
  {
    weights[level] = weight;
  }
  return weights;
}

Real peakBaseSpeed(const TargetParams& target)
{
  return peakSpeed(
      target.quantity, "poiseuille", target.bulk_speed, 1.0, 1.5);
}

Real maxPulseSpeed(const TargetParams& target)
{
  return peakBaseSpeed(target) * (1.0 + std::abs(target.pulse_amplitude));
}

Vector<Real> makeTrueParams(const MixedFESpace&     space,
                            const DirichletControl& control,
                            const TargetParams&     target,
                            const Vector<Real>&     times)
{
  Vector<Real> prm(times.size() * control.numDofs());

  for (Index level = 0; level < times.size(); ++level)
  {
    for (Index i = 0; i < control.numDofs(); ++i)
    {
      prm[level * control.numDofs() + i] =
          profileVelocityValue(space, control, target, times[level], i);
    }
  }
  return prm;
}

std::optional<Real> componentValue(const BCsParams& bc, Index component)
{
  if (component == 0 && bc.ux)
  {
    return *bc.ux;
  }
  if (component == 1 && bc.uy)
  {
    return *bc.uy;
  }
  if (component == 2 && bc.uz)
  {
    return *bc.uz;
  }
  return std::nullopt;
}

Vector<Real> initialControlParams(const MixedFESpace&     space,
                                  const DirichletControl& ctr,
                                  const BCsParams&        bc,
                                  const Vector<Real>&     times)
{
  if (times.empty())
  {
    throw std::runtime_error("initialControlParams requires control times");
  }

  if (bc.vel)
  {
    return makeTrueParams(space, ctr, *bc.vel, times);
  }

  Vector<Real> prm(times.size() * ctr.numDofs());
  prm.setZero();

  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  for (Index level = 0; level < times.size(); ++level)
  {
    for (Index i = 0; i < ctr.numDofs(); ++i)
    {
      const Index dof       = ctr.stateDof(i);
      const Index component = dof % nd;
      if (const auto value = componentValue(bc, component))
      {
        prm[level * ctr.numDofs() + i] = *value;
      }
    }
  }
  return prm;
}

Vector<Real> initialInverseParams(const MixedFESpace&           space,
                                  const DirichletControl&       control,
                                  const Params&                 prm,
                                  const InverseParameterLayout& layout,
                                  Index                         steps,
                                  Real                          dt,
                                  const Vector<Real>&           ctr_times)
{
  (void) steps;
  (void) dt;
  Vector<Real> out(layout.total_size);
  out.setZero();

  const Vector<Real> ctr_prm =
      initialControlParams(
          space,
          control,
          controlBoundary(prm),
          ctr_times);
  if (ctr_prm.size() != layout.ctr_size)
  {
    throw std::runtime_error(
        "initialInverseParams control parameter size mismatch");
  }
  for (Index i = 0; i < ctr_prm.size(); ++i)
  {
    out[layout.ctr_offset + i] = ctr_prm[i];
  }
  return out;
}

void setControlParams(const InverseParameterLayout& layout,
                      const Vector<Real>&           ctr_prm,
                      Vector<Real>&                 prm)
{
  if (prm.size() != layout.total_size
      || ctr_prm.size() != layout.ctr_size)
  {
    throw std::runtime_error("setControlParams size mismatch");
  }

  for (Index i = 0; i < layout.ctr_size; ++i)
  {
    prm[layout.ctr_offset + i] = ctr_prm[i];
  }
}

Vector<Real> controlParams(const InverseParameterLayout& layout,
                           const Vector<Real>&           prm)
{
  if (prm.size() != layout.total_size)
  {
    throw std::runtime_error("controlParams parameter size mismatch");
  }

  Vector<Real> out(layout.ctr_size);
  for (Index i = 0; i < layout.ctr_size; ++i)
  {
    out[i] = prm[layout.ctr_offset + i];
  }
  return out;
}

Vector<Real> optimizerScale(const InverseParameterLayout& layout,
                            const OptimizerParams::Scale& scale)
{
  Vector<Real> out(layout.total_size);
  for (Index i = 0; i < layout.init_vel_size; ++i)
  {
    out[layout.init_vel_offset + i] = scale.initial_velocity;
  }
  for (Index i = 0; i < layout.ctr_size; ++i)
  {
    out[layout.ctr_offset + i] = scale.boundary;
  }
  return out;
}

void setInitialVelocityParams(const Vector<Index>&          velocity_dofs,
                              const InverseParameterLayout& layout,
                              const Vector<Real>&           state,
                              Vector<Real>&                 prm)
{
  if (prm.size() != layout.total_size
      || velocity_dofs.size() != layout.init_vel_size)
  {
    throw std::runtime_error("setInitialVelocityParams size mismatch");
  }

  for (Index i = 0; i < velocity_dofs.size(); ++i)
  {
    const Index dof = velocity_dofs[i];
    if (dof < 0 || dof >= state.size())
    {
      throw std::runtime_error(
          "setInitialVelocityParams velocity dof is out of range");
    }
    prm[layout.init_vel_offset + i] = state[dof];
  }
}

void initializeOptimizationGuess(const MixedFESpace&           space,
                                 const DirichletControl&       control,
                                 const Params&                 prm,
                                 const InverseParameterLayout& layout,
                                 const Vector<Index>&          vdofs,
                                 TimeLinearStateSolver&        state_solver,
                                 const Vector<Real>&           ctr_times,
                                 Vector<Real>&                 prm_init,
                                 Vector<Real>*                 x0)
{
  if (prm_init.size() != layout.total_size)
  {
    throw std::runtime_error(
        "initializeOptimizationGuess parameter size mismatch");
  }

  const Vector<Real> ctr_prm =
      initialControlParams(space, control, controlBoundary(prm), ctr_times);
  setControlParams(layout, ctr_prm, prm_init);

  Vector<Real> guess_prm(layout.total_size);
  guess_prm.setZero();
  setControlParams(layout, ctr_prm, guess_prm);

  TimeTrajectory tr;
  state_solver.solve(guess_prm, tr);

  const Vector<Real>& final_state = tr[tr.numSteps()];
  if (x0 != nullptr)
  {
    *x0 = final_state;
  }
  setInitialVelocityParams(vdofs, layout, final_state, prm_init);
}

void initialStateFromParams(const Vector<Index>&          velocity_dofs,
                            const InverseParameterLayout& layout,
                            Index                         num_states,
                            const Vector<Real>&           prm,
                            Vector<Real>&                 out)
{
  Vector<Real> base_state(num_states);
  base_state.setZero();
  initialStateFromParams(velocity_dofs, layout, base_state, prm, out);
}

void initialStateFromParams(const Vector<Index>&          velocity_dofs,
                            const InverseParameterLayout& layout,
                            const Vector<Real>&           base_state,
                            const Vector<Real>&           prm,
                            Vector<Real>&                 out)
{
  if (prm.size() != layout.total_size)
  {
    throw std::runtime_error(
        "initialStateFromParams parameter size mismatch");
  }
  if (velocity_dofs.size() != layout.init_vel_size)
  {
    throw std::runtime_error(
        "initialStateFromParams velocity dof size mismatch");
  }

  if (base_state.empty())
  {
    throw std::runtime_error(
        "initialStateFromParams base state is empty");
  }
  out = base_state;
  for (Index i = 0; i < velocity_dofs.size(); ++i)
  {
    const Index dof = velocity_dofs[i];
    if (dof < 0 || dof >= out.size())
    {
      throw std::runtime_error(
          "initialStateFromParams velocity dof is out of range");
    }
    out[dof] = prm[layout.init_vel_offset + i];
  }
}

void applyInitialVelocityParamJacT(
    const Vector<Index>&          velocity_dofs,
    const InverseParameterLayout& layout,
    const Vector<Real>&           state_grad,
    Vector<Real>&                 out)
{
  if (velocity_dofs.size() != layout.init_vel_size)
  {
    throw std::runtime_error(
        "applyInitialVelocityParamJacT velocity dof size mismatch");
  }

  resize(out, layout.total_size);
  for (Index i = 0; i < velocity_dofs.size(); ++i)
  {
    const Index dof = velocity_dofs[i];
    if (dof < 0 || dof >= state_grad.size())
    {
      throw std::runtime_error(
          "applyInitialVelocityParamJacT velocity dof is out of range");
    }
    out[layout.init_vel_offset + i] = state_grad[dof];
  }
}

void controlBounds(const MixedFESpace&     space,
                   const DirichletControl& control,
                   const TargetParams&     target,
                   const BoundsParams&     bounds,
                   Index                   steps,
                   Vector<Real>&           lower,
                   Vector<Real>&           upper)
{
  lower.resize(control.numParams(steps));
  upper.resize(control.numParams(steps));

  const auto  u_dof  = space.field(0);
  const Index nd     = u_dof.numComponents();
  const auto  normal = unit(target.normal);
  const Real  max_axial =
      bounds.max ? *bounds.max : bounds.max_scale * maxPulseSpeed(target);

  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < control.numDofs(); ++i)
    {
      const Index dof  = control.stateDof(i);
      const Index comp = dof % nd;
      const Index idx  = control.paramIndex(step, i);

      const Real normal_comp = normal[static_cast<std::size_t>(comp)];
      const Real lo          = bounds.min * normal_comp;
      const Real hi          = max_axial * normal_comp;
      lower[idx]             = std::min(lo, hi);
      upper[idx]             = std::max(lo, hi);
    }
  }
}

void controlBounds(const MixedFESpace&     space,
                   const DirichletControl& control,
                   const BoundsParams&     bounds,
                   Index                   steps,
                   Vector<Real>&           lower,
                   Vector<Real>&           upper)
{
  if (!bounds.max)
  {
    throw std::runtime_error("controlBounds requires inverse.bounds.max");
  }

  lower.resize(control.numParams(steps));
  upper.resize(control.numParams(steps));

  const auto  u_dof  = space.field(0);
  const Index nd     = u_dof.numComponents();
  const auto  normal = unit(bounds.normal);

  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < control.numDofs(); ++i)
    {
      const Index dof  = control.stateDof(i);
      const Index comp = dof % nd;
      const Index idx  = control.paramIndex(step, i);

      const Real normal_comp = normal[static_cast<std::size_t>(comp)];
      const Real lo          = bounds.min * normal_comp;
      const Real hi          = *bounds.max * normal_comp;
      lower[idx]             = std::min(lo, hi);
      upper[idx]             = std::max(lo, hi);
    }
  }
}

void inverseBounds(const MixedFESpace&           space,
                   const DirichletControl&       control,
                   const Params&                 prm,
                   const InverseParameterLayout& layout,
                   Index                         steps,
                   Vector<Real>&                 lower,
                   Vector<Real>&                 upper)
{
  resize(lower, layout.total_size);
  resize(upper, layout.total_size);

  if (layout.hasInitialVelocity())
  {
    constexpr Real unbounded = 1.0e30;
    const Real     lo =
        prm.inv.init_vel.lower.value_or(-unbounded);
    const Real hi =
        prm.inv.init_vel.upper.value_or(unbounded);
    for (Index i = 0; i < layout.init_vel_size; ++i)
    {
      lower[layout.init_vel_offset + i] = lo;
      upper[layout.init_vel_offset + i] = hi;
    }
  }

  Vector<Real> ctr_lower;
  Vector<Real> ctr_upper;
  if (prm.inv.bounds.max)
  {
    controlBounds(
        space,
        control,
        prm.inv.bounds,
        layout.ctr_levels,
        ctr_lower,
        ctr_upper);
  }
  else
  {
    controlBounds(space,
                  control,
                  controlTarget(prm),
                  prm.inv.bounds,
                  layout.ctr_levels,
                  ctr_lower,
                  ctr_upper);
  }
  if (ctr_lower.size() != layout.ctr_size
      || ctr_upper.size() != layout.ctr_size)
  {
    throw std::runtime_error(
        "inverseBounds control bound size mismatch");
  }
  for (Index i = 0; i < layout.ctr_size; ++i)
  {
    lower[layout.ctr_offset + i] = ctr_lower[i];
    upper[layout.ctr_offset + i] = ctr_upper[i];
  }
}

void checkInverseRunParams(const Params& prm)
{
  if (prm.fwd.time.steps <= 0 || prm.fwd.time.dt <= 0.0)
  {
    throw std::runtime_error("time steps and dt must be positive");
  }
  if (prm.inv.obs.file.empty())
  {
    throw std::runtime_error("inverse.obs.file is required");
  }
}

AppNsVar::AppNsVar(const Params& prm)
  : steps(observationSolveSteps(prm)),
    dt(prm.fwd.time.dt),
    mesh(GmshReader::read(prm.fwd.mesh.file)),
    elem(makeElement(mesh)),
    space(makeSpace(mesh, *elem)),
    fluid(fluidParams(prm)),
    quad(makeVelocityQuadrature(space)),
    ns(makeAppKernel(space, quad, fluid, dt)),
    fem(steps,
        DofLayout(space),
        DofLayout(space),
        DofLayout(space),
        ns),
    bc(controlBoundary(prm)),
    ctr(makeBoundaryControl(space, prm, bc)),
    init_vdofs(
        prm.inv.init_vel.enabled ? initialvdofs(space, prm, ctr)
                                 : Vector<Index>{}),
    obs_data(problem::readTimeObsData(prm.inv.obs.file)),
    ctr_times(controlKnotTimes(obs_data, dt, steps)),
    ctr_time_stencils(controlTimeStencils(steps, dt, ctr_times)),
    layout(inverseParameterLayout(space,
                                  ctr,
                                  prm.inv.init_vel,
                                  steps,
                                  init_vdofs.size(),
                                  ctr_times.size())),
    fixed(fixedDofValues(space, prm, ctr, steps, dt)),
    eq(fem,
       ctr,
       fixed.dofs,
       layout.ctr_offset,
       layout.total_size,
       fixed.values,
       ctr_time_stencils),
    x0(initialVelocityBoundaryState(space, prm, ctr)),
    pattern(assembly::SparsityPatternBuilder::build(space)),
    prm0(initialInverseParams(
        space, ctr, prm, layout, steps, dt, ctr_times))
{
  if (prm0.size() != eq.numParams())
  {
    throw std::runtime_error("initial inverse parameter size mismatch");
  }
}

Objective::Objective(const Params&      prm,
                     const AppNsVar& core)
  : data(observationDataOnSolveLevels(core.obs_data, core.dt, core.steps)),
    op(core.steps,
       core.space,
       0,
       data.points(),
       data.components(),
       core.eq.numParams()),
    err(op,
        data,
        wError(core.steps, prm.inv.alpha, false),
        core.dt,
        0.0),
    ctr_reg(core.steps,
            core.eq.numStates(),
            core.layout.ctr_levels,
            core.ctr.numDofs(),
            prm.inv.reg.beta1,
            prm.inv.reg.beta2),
    reg(ctr_reg, core.eq.numParams(), core.layout.ctr_offset),
    init_reg(core.steps,
             core.eq.numStates(),
             core.layout,
             prm.inv.reg.beta4),
    space_reg(core.steps,
              core.eq.numStates(),
              core.eq.numParams(),
              core.layout.ctr_offset,
              core.layout.ctr_levels,
              core.ctr.numDofs(),
              controlSpatialEdges(core.space, prm.inv.ctr, core.ctr),
              prm.inv.reg.beta3),
    obj(core.steps, core.eq.numStates(), core.eq.numParams())
{
  obj.add(err).add(reg).add(init_reg).add(space_reg);
}

} // namespace femx::navier_var_new
