#include "RunSupport.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

#include <femx/fem/VelocityProfile.hpp>
#include <femx/core/Math.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/fem/elements/LagrangeTriangleP1.hpp>
#include <femx/fem/ObservationGrid.hpp>
#include <femx/fem/TimePointSampler.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>

using namespace femx::fem;
using namespace femx::problem;
using namespace femx::solve;

namespace femx::navier_var
{

namespace
{

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
      TimePointSampler::filterPointsInside(space, 0, points);
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

bool hasFixedVelocityValue(const BCsParams& bc)
{
  return bc.ux || bc.uy || bc.uz || bc.velocity;
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
  for (const auto& bc : prm.forward.bcs)
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

void addProfileVelocityValues(const MixedFESpace&            space,
                              const DirichletControl&        control,
                              const BCsParams&               bc,
                              Index                          steps,
                              Real                           dt,
                              bool                           pre_observation_initial,
                              std::map<Index, Vector<Real>>& values)
{
  if (!bc.velocity)
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
                    trueValue(space,
                              control,
                              *bc.velocity,
                              step,
                              i,
                              dt,
                              pre_observation_initial));
    }
  }
}

Vector<Index> fixedVelocityDofsForBc(const MixedFESpace& space,
                                     const BCsParams&    bc)
{
  const DirichletControl boundary =
      makeVelocityControl(space, bcSelector(bc));
  if (bc.velocity)
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
  if (bc.velocity)
  {
    addInitialProfileVelocityValues(
        space, control, *bc.velocity, 0.0, values);
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

void splitState(const MixedFESpace& space,
                const Vector<Real>& x,
                Vector<Real>&       ux,
                Vector<Real>&       uy,
                Vector<Real>&       uz,
                Vector<Real>&       p)
{
  const Index nodes = space.mesh().numNodes();
  resize(ux, nodes);
  resize(uy, nodes);
  resize(uz, nodes);
  resize(p, nodes);

  const auto  u_dof = space.field(0);
  const auto  p_dof = space.field(1);
  const Index nc    = u_dof.numComponents();
  for (Index in = 0; in < nodes; ++in)
  {
    ux[in] = x[u_dof.globalDof(in, 0)];
    uy[in] = nc > 1 ? x[u_dof.globalDof(in, 1)] : 0.0;
    uz[in] = nc > 2 ? x[u_dof.globalDof(in, 2)] : 0.0;
    p[in]  = x[p_dof.globalDof(in, 0)];
  }
}

void assignControlDof(const MixedFESpace& space,
                      Index               dof,
                      Real                value,
                      Vector<Real>&       ux,
                      Vector<Real>&       uy,
                      Vector<Real>&       uz)
{
  const auto u_dof = space.field(0);
  for (Index node = 0; node < space.mesh().numNodes(); ++node)
  {
    for (Index d = 0; d < u_dof.numComponents(); ++d)
    {
      if (u_dof.globalDof(node, d) != dof)
      {
        continue;
      }
      if (d == 0)
      {
        ux[node] = value;
      }
      else if (d == 1)
      {
        uy[node] = value;
      }
      else if (d == 2)
      {
        uz[node] = value;
      }
      return;
    }
  }
}

void controlField(const MixedFESpace&     space,
                  const DirichletControl& control,
                  const Vector<Real>&     prm,
                  Index                   step,
                  Vector<Real>&           ux,
                  Vector<Real>&           uy,
                  Vector<Real>&           uz)
{
  const Index nodes = space.mesh().numNodes();
  resize(ux, nodes);
  resize(uy, nodes);
  resize(uz, nodes);
  if (step < 0)
  {
    return;
  }

  for (Index i = 0; i < control.numDofs(); ++i)
  {
    assignControlDof(space,
                     control.stateDof(i),
                     prm[control.paramIndex(step, i)],
                     ux,
                     uy,
                     uz);
  }
}

void ensureDir(const std::string& basename)
{
  const std::filesystem::path path(basename);
  const std::filesystem::path dir = path.parent_path();
  if (!dir.empty())
  {
    std::filesystem::create_directories(dir);
  }
}

} // namespace

InitialVelocityStateSolver::InitialVelocityStateSolver(
    TimeMatrixLinearStateSolver& solver,
    Vector<Index>                velocity_dofs,
    InverseParameterLayout       layout,
    Vector<Real>                 base_initial_state)
  : solver_(solver),
    velocity_dofs_(std::move(velocity_dofs)),
    layout_(layout),
    base_initial_state_(std::move(base_initial_state))
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
  if (velocity_dofs_.size() != layout_.initial_velocity_size)
  {
    throw std::runtime_error(
        "InitialVelocityStateSolver velocity dof size mismatch");
  }
  if (base_initial_state_.empty())
  {
    base_initial_state_.resize(solver_.numStates());
  }
  else if (base_initial_state_.size() != solver_.numStates())
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

void InitialVelocityStateSolver::solve(const Vector<Real>&  prm,
                                       TimeStateTrajectory& tr)
{
  initialStateFromParams(
      velocity_dofs_, layout_, base_initial_state_, prm, initial_state_);
  solver_.setInitialState(initial_state_);
  solver_.solve(prm, tr);
}

ParameterSliceTimeObjective::ParameterSliceTimeObjective(
    const TimeObjectiveFunctional& base,
    Index                          total_params,
    Index                          offset)
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

Real ParameterSliceTimeObjective::value(const TimeStateTrajectory& tr,
                                        const Vector<Real>&        prm) const
{
  return base_.value(tr, slice(prm));
}

void ParameterSliceTimeObjective::stateGrad(Index                      level,
                                            const TimeStateTrajectory& tr,
                                            const Vector<Real>&        prm,
                                            Vector<Real>&              out) const
{
  base_.stateGrad(level, tr, slice(prm), out);
}

void ParameterSliceTimeObjective::paramGrad(const TimeStateTrajectory& tr,
                                            const Vector<Real>&        prm,
                                            Vector<Real>&              out) const
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
      || layout_.initial_velocity_offset < 0
      || layout_.initial_velocity_size < 0
      || layout_.initial_velocity_offset + layout_.initial_velocity_size
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

Real InitialVelocityRegularization::value(const TimeStateTrajectory& tr,
                                          const Vector<Real>&        prm) const
{
  (void) tr;
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "InitialVelocityRegularization parameter size mismatch");
  }

  Real out = 0.0;
  for (Index i = 0; i < layout_.initial_velocity_size; ++i)
  {
    const Real value  = prm[layout_.initial_velocity_offset + i];
    out              += 0.5 * beta_ * value * value;
  }
  return out;
}

void InitialVelocityRegularization::stateGrad(
    Index                      level,
    const TimeStateTrajectory& tr,
    const Vector<Real>&        prm,
    Vector<Real>&              out) const
{
  (void) level;
  (void) tr;
  (void) prm;
  resize(out, num_states_);
}

void InitialVelocityRegularization::paramGrad(
    const TimeStateTrajectory& tr,
    const Vector<Real>&        prm,
    Vector<Real>&              out) const
{
  (void) tr;
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "InitialVelocityRegularization parameter size mismatch");
  }

  resize(out, numParams());
  for (Index i = 0; i < layout_.initial_velocity_size; ++i)
  {
    const Index idx = layout_.initial_velocity_offset + i;
    out[idx]        = beta_ * prm[idx];
  }
}

std::unique_ptr<FiniteElement> makeElement(const Mesh& mesh)
{
  if (mesh.numElems() == 0)
  {
    throw std::runtime_error("Mesh has no cells");
  }

  const Cell::Shape shape = mesh.cells().front().shape();
  if (shape == Cell::Shape::Quadrilateral)
  {
    return std::make_unique<LagrangeQuadQ1>();
  }
  if (shape == Cell::Shape::Triangle)
  {
    return std::make_unique<LagrangeTriangleP1>();
  }
  if (shape == Cell::Shape::Tetrahedron)
  {
    return std::make_unique<LagrangeTetrahedronP1>();
  }
  throw std::runtime_error("Unsupported mesh cell type for ns-var");
}

MixedFESpace makeSpace(Mesh& mesh, FiniteElement& elem)
{
  FESpace u_space(&mesh, &elem, mesh.dim());
  FESpace p_space(&mesh, &elem);

  MixedFESpace space;
  space.addField(u_space);
  space.addField(p_space);
  space.setup();
  return space;
}

Point3 selectorCenter(const Mesh& mesh, const BoundarySelector& selector)
{
  if (!selector.name.empty())
  {
    return boundaryCenter(mesh, selector.name);
  }
  return boundaryCenter(mesh, selector.physical);
}

Vector<Index> gaugeDofs(const MixedFESpace&     space,
                        const BoundarySelector& selector)
{
  Index      node_out = 0;
  Real       dist_out = 0.0;
  const auto center   = selectorCenter(space.mesh(), selector);
  for (Index node = 0; node < space.mesh().numNodes(); ++node)
  {
    const Real dist = sqDist(space.mesh().node(node), center);
    if (node == 0 || dist < dist_out)
    {
      node_out = node;
      dist_out = dist;
    }
  }

  Vector<Index> dofs(1);
  dofs[0] = space.field(1).globalDof(node_out, 0);
  return dofs;
}

DirichletControl makeVelocityControl(
    const MixedFESpace&     space,
    const BoundarySelector& selector)
{
  if (!selector.name.empty())
  {
    return femx::makeVelocityControl(space, selector.name);
  }
  return femx::makeVelocityControl(space, selector.physical);
}

Vector<Index> initialVelocityDofs(const MixedFESpace& space)
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

Vector<Index> initialVelocityDofs(const MixedFESpace&     space,
                                  const Params&           prm,
                                  const DirichletControl& control)
{
  Vector<Index> constrained = control.stateDofs();
  for (const auto& bc : prm.forward.bcs)
  {
    if (matchesBoundaryConfig(prm.inverse.control, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }
    appendUniqueExcept(
        constrained, fixedVelocityDofsForBc(space, bc), control.stateDofs());
  }

  const Vector<Index> all = initialVelocityDofs(space);
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

  for (const auto& bc : prm.forward.bcs)
  {
    if (matchesBoundaryConfig(prm.inverse.control, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }

    const Vector<Index> fixed_dofs = fixedVelocityDofsForBc(space, bc);
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
    Index                        steps)
{
  if (steps <= 0)
  {
    throw std::runtime_error(
        "inverseParameterLayout requires positive steps");
  }

  return inverseParameterLayout(space,
                                control,
                                initial_velocity,
                                steps,
                                initialVelocityDofs(space).size());
}

InverseParameterLayout inverseParameterLayout(
    const MixedFESpace&          space,
    const DirichletControl&      control,
    const InitialVelocityParams& initial_velocity,
    Index                        steps,
    Index                        initial_velocity_size)
{
  (void) space;
  if (steps <= 0 || initial_velocity_size < 0)
  {
    throw std::runtime_error(
        "inverseParameterLayout received invalid sizes");
  }

  InverseParameterLayout layout;
  if (initial_velocity.enabled)
  {
    layout.initial_velocity_offset = 0;
    layout.initial_velocity_size   = initial_velocity_size;
  }
  layout.control_offset = layout.initial_velocity_offset
                          + layout.initial_velocity_size;
  layout.control_size = control.numParams(steps);
  layout.total_size   = layout.control_offset + layout.control_size;
  return layout;
}

Vector<Index> fixedDofs(const MixedFESpace&     space,
                        const Params&           prm,
                        const DirichletControl& control)
{
  Vector<Index> dofs =
      gaugeDofs(space, bcSelector(pressureGaugeBoundary(prm)));

  for (const auto& bc : prm.forward.bcs)
  {
    if (matchesBoundaryConfig(prm.inverse.control, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }
    appendUniqueExcept(
        dofs, fixedVelocityDofsForBc(space, bc), control.stateDofs());
  }
  return dofs;
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

  for (const auto& bc : prm.forward.bcs)
  {
    if (matchesBoundaryConfig(prm.inverse.control, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }

    const Vector<Index> fixed_dofs = fixedVelocityDofsForBc(space, bc);
    Vector<Index> active_dofs;
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
                             prm.inverse.initial_velocity.enabled,
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

  return std::make_unique<TimePointSampler>(
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

std::unique_ptr<TimeObservationOperator> makeObsFromData(
    const MixedFESpace&        space,
    const TimeObservationData& data,
    Index                      steps,
    Index                      num_states,
    Index                      num_prm)
{
  if (!data.hasLayout())
  {
    throw std::runtime_error(
        "Observation data file has no point layout; regenerate it with make-obs");
  }
  if (data.sampler() != "point")
  {
    throw std::runtime_error(
        "Unsupported observation sampler in data file: " + data.sampler());
  }
  if (num_states != space.numDofs())
  {
    throw std::runtime_error(
        "makeObsFromData state size does not match FEM space");
  }
  return std::make_unique<TimePointSampler>(
      steps, space, 0, data.points(), data.components(), num_prm);
}

Vector<Real> misfitW(Index num_steps,
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

AxialVelocityProfile targetProfile(const TargetParams& target)
{
  return poiseuilleProfile(target.center, target.normal, target.radius);
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

Real trueValue(const MixedFESpace&     space,
               const DirichletControl& control,
               const TargetParams&     target,
               Index                   step,
               Index                   i,
               Real                    dt,
               bool                    pre_observation_initial)
{
  const Real t =
      static_cast<Real>(pre_observation_initial ? step : step + 1) * dt;
  return profileVelocityValue(space, control, target, t, i);
}

Vector<Real> makeTrueParams(const MixedFESpace&     space,
                            const DirichletControl& control,
                            const TargetParams&     target,
                            Index                   steps,
                            Real                    dt,
                            bool                    pre_observation_initial)
{
  Vector<Real> prm(control.numParams(steps));

  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < control.numDofs(); ++i)
    {
      prm[control.paramIndex(step, i)] =
          trueValue(space,
                    control,
                    target,
                    step,
                    i,
                    dt,
                    pre_observation_initial);
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
                                  const DirichletControl& control,
                                  const BCsParams&        bc,
                                  Index                   steps,
                                  Real                    dt,
                                  bool                    pre_observation_initial)
{
  if (steps <= 0)
  {
    throw std::runtime_error("initialControlParams requires positive steps");
  }

  if (bc.velocity)
  {
    return makeTrueParams(
        space, control, *bc.velocity, steps, dt, pre_observation_initial);
  }

  Vector<Real> prm(control.numParams(steps));
  prm.setZero();

  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < control.numDofs(); ++i)
    {
      const Index dof       = control.stateDof(i);
      const Index component = dof % nd;
      if (const auto value = componentValue(bc, component))
      {
        prm[control.paramIndex(step, i)] = *value;
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
                                  Real                          dt)
{
  Vector<Real> out(layout.total_size);
  out.setZero();

  const Vector<Real> control_prm =
      initialControlParams(
          space,
          control,
          controlBoundary(prm),
          steps,
          dt,
          prm.inverse.initial_velocity.enabled);
  if (control_prm.size() != layout.control_size)
  {
    throw std::runtime_error(
        "initialInverseParams control parameter size mismatch");
  }
  for (Index i = 0; i < control_prm.size(); ++i)
  {
    out[layout.control_offset + i] = control_prm[i];
  }
  return out;
}

Real inferredControlRadius(const MixedFESpace&     space,
                           const DirichletControl& control,
                           const Point3&           center,
                           const Point3&           normal)
{
  const auto    u_dof = space.field(0);
  Vector<Index> nodes;
  for (Index i = 0; i < control.numDofs(); ++i)
  {
    const Index node = control.stateDof(i) / u_dof.numComponents();
    if (!contains(nodes, node))
    {
      nodes.push_back(node);
    }
  }

  Real radius2 = 0.0;
  for (Index node : nodes)
  {
    radius2 =
        std::max(radius2,
                 radialSq(space.mesh().node(node), center, normal));
  }
  if (radius2 <= 0.0)
  {
    throw std::runtime_error(
        "Could not infer a positive Poiseuille seed radius from the control boundary");
  }
  return std::sqrt(radius2);
}

Real clippedInitialVelocityValue(const InitialVelocityParams& prm,
                                 Real                         value)
{
  if (prm.lower)
  {
    value = std::max(value, *prm.lower);
  }
  if (prm.upper)
  {
    value = std::min(value, *prm.upper);
  }
  return value;
}

TargetParams constantPoiseuilleSeedTarget(const MixedFESpace&     space,
                                          const DirichletControl& control,
                                          const Params&           prm)
{
  const BCsParams& control_bc = controlBoundary(prm);
  if (control_bc.velocity)
  {
    TargetParams target    = *control_bc.velocity;
    target.pulse_amplitude = 0.0;
    return target;
  }

  if (!prm.inverse.bounds.axial_max)
  {
    throw std::runtime_error(
        "constant Poiseuille seed requires inverse.bounds.axial_max");
  }

  TargetParams target;
  target.quantity        = "max_velocity";
  target.bulk_speed      = *prm.inverse.bounds.axial_max;
  target.pulse_amplitude = 0.0;
  target.period          = 1.0;
  target.center          = selectorCenter(space.mesh(), prm.inverse.control);
  target.normal          = prm.inverse.bounds.normal;
  target.radius =
      inferredControlRadius(space, control, target.center, target.normal);
  return target;
}

Vector<Real> constantPoiseuilleSeedParams(const MixedFESpace&     space,
                                          const DirichletControl& control,
                                          const Params&           prm,
                                          Index                   steps,
                                          Real                    dt)
{
  const TargetParams target =
      constantPoiseuilleSeedTarget(space, control, prm);
  return makeTrueParams(
      space,
      control,
      target,
      steps,
      dt,
      prm.inverse.initial_velocity.enabled);
}

void seedControlParams(const InverseParameterLayout& layout,
                       const Vector<Real>&           control_prm,
                       Vector<Real>&                 prm)
{
  if (prm.size() != layout.total_size
      || control_prm.size() != layout.control_size)
  {
    throw std::runtime_error("seedControlParams size mismatch");
  }

  for (Index i = 0; i < layout.control_size; ++i)
  {
    prm[layout.control_offset + i] = control_prm[i];
  }
}

void seedInitialVelocityParamsFromState(
    const Vector<Index>&          velocity_dofs,
    const InverseParameterLayout& layout,
    const InitialVelocityParams&  initial_velocity,
    const Vector<Real>&           state,
    Vector<Real>&                 prm)
{
  if (prm.size() != layout.total_size
      || velocity_dofs.size() != layout.initial_velocity_size)
  {
    throw std::runtime_error(
        "seedInitialVelocityParamsFromState size mismatch");
  }

  for (Index i = 0; i < velocity_dofs.size(); ++i)
  {
    const Index dof = velocity_dofs[i];
    if (dof < 0 || dof >= state.size())
    {
      throw std::runtime_error(
          "seedInitialVelocityParamsFromState velocity dof is out of range");
    }
    prm[layout.initial_velocity_offset + i] =
        clippedInitialVelocityValue(initial_velocity, state[dof]);
  }
}

void seedInitialVelocityFromConstantPoiseuille(
    const MixedFESpace&              space,
    const DirichletControl&          control,
    const Params&                    prm,
    const InverseParameterLayout&    layout,
    const Vector<Index>&             velocity_dofs,
    TimeMatrixLinearStateSolver&     state_solver,
    Vector<Real>&                    prm_init)
{
  if (prm_init.size() != layout.total_size)
  {
    throw std::runtime_error(
        "seedInitialVelocityFromConstantPoiseuille parameter size mismatch");
  }

  const Vector<Real> control_prm =
      constantPoiseuilleSeedParams(
          space, control, prm, state_solver.numSteps(), prm.forward.time.dt);
  seedControlParams(layout, control_prm, prm_init);
  if (!layout.hasInitialVelocity())
  {
    return;
  }

  Vector<Real> seed_prm(layout.total_size);
  seed_prm.setZero();
  seedControlParams(layout, control_prm, seed_prm);

  TimeStateTrajectory seed_tr;
  state_solver.solve(seed_prm, seed_tr);
  seedInitialVelocityParamsFromState(velocity_dofs,
                                     layout,
                                     prm.inverse.initial_velocity,
                                     seed_tr[seed_tr.numSteps()],
                                     prm_init);
}

void initialStateFromParams(const Vector<Index>&          velocity_dofs,
                            const InverseParameterLayout& layout,
                            Index                         num_states,
                            const Vector<Real>&           prm,
                            Vector<Real>&                 out)
{
  Vector<Real> base_state(num_states);
  base_state.setZero();
  initialStateFromParams(
      velocity_dofs, layout, base_state, prm, out);
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
  if (velocity_dofs.size() != layout.initial_velocity_size)
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
    out[dof] = prm[layout.initial_velocity_offset + i];
  }
}

void applyInitialVelocityParamJacT(
    const Vector<Index>&          velocity_dofs,
    const InverseParameterLayout& layout,
    const Vector<Real>&           state_grad,
    Vector<Real>&                 out)
{
  if (velocity_dofs.size() != layout.initial_velocity_size)
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
    out[layout.initial_velocity_offset + i] = state_grad[dof];
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

  const auto  u_dof           = space.field(0);
  const Index nd              = u_dof.numComponents();
  const auto  normal          = unit(target.normal);
  Real        max_normal_comp = 0.0;
  for (Index d = 0; d < nd; ++d)
  {
    max_normal_comp =
        std::max(max_normal_comp,
                 std::abs(normal[static_cast<std::size_t>(d)]));
  }
  const Real max_axial =
      bounds.axial_max ? *bounds.axial_max
                       : bounds.axial_max_scale * maxPulseSpeed(target);

  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < control.numDofs(); ++i)
    {
      const Index dof  = control.stateDof(i);
      const Index comp = dof % nd;
      const Index idx  = control.paramIndex(step, i);

      const Real normal_comp = normal[static_cast<std::size_t>(comp)];
      const bool axial       = std::abs(normal_comp) >= 0.5 * max_normal_comp;
      if (axial || !bounds.fix_non_axial)
      {
        const Real lo = bounds.axial_min * normal_comp;
        const Real hi = max_axial * normal_comp;
        lower[idx]    = std::min(lo, hi);
        upper[idx]    = std::max(lo, hi);
      }
      else
      {
        lower[idx] = 0.0;
        upper[idx] = 0.0;
      }
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
  if (!bounds.axial_max)
  {
    throw std::runtime_error("controlBounds requires inverse.bounds.axial_max");
  }

  lower.resize(control.numParams(steps));
  upper.resize(control.numParams(steps));

  const auto  u_dof           = space.field(0);
  const Index nd              = u_dof.numComponents();
  const auto  normal          = unit(bounds.normal);
  Real        max_normal_comp = 0.0;
  for (Index d = 0; d < nd; ++d)
  {
    max_normal_comp =
        std::max(max_normal_comp,
                 std::abs(normal[static_cast<std::size_t>(d)]));
  }

  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < control.numDofs(); ++i)
    {
      const Index dof  = control.stateDof(i);
      const Index comp = dof % nd;
      const Index idx  = control.paramIndex(step, i);

      const Real normal_comp = normal[static_cast<std::size_t>(comp)];
      const bool axial       = std::abs(normal_comp) >= 0.5 * max_normal_comp;
      if (axial || !bounds.fix_non_axial)
      {
        const Real lo = bounds.axial_min * normal_comp;
        const Real hi = *bounds.axial_max * normal_comp;
        lower[idx]    = std::min(lo, hi);
        upper[idx]    = std::max(lo, hi);
      }
      else
      {
        lower[idx] = 0.0;
        upper[idx] = 0.0;
      }
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
        prm.inverse.initial_velocity.lower.value_or(-unbounded);
    const Real hi =
        prm.inverse.initial_velocity.upper.value_or(unbounded);
    for (Index i = 0; i < layout.initial_velocity_size; ++i)
    {
      lower[layout.initial_velocity_offset + i] = lo;
      upper[layout.initial_velocity_offset + i] = hi;
    }
  }

  Vector<Real> control_lower;
  Vector<Real> control_upper;
  if (prm.inverse.bounds.axial_max)
  {
    controlBounds(
        space, control, prm.inverse.bounds, steps, control_lower, control_upper);
  }
  else
  {
    controlBounds(space,
                  control,
                  controlTarget(prm),
                  prm.inverse.bounds,
                  steps,
                  control_lower,
                  control_upper);
  }
  if (control_lower.size() != layout.control_size
      || control_upper.size() != layout.control_size)
  {
    throw std::runtime_error(
        "inverseBounds control bound size mismatch");
  }
  for (Index i = 0; i < layout.control_size; ++i)
  {
    lower[layout.control_offset + i] = control_lower[i];
    upper[layout.control_offset + i] = control_upper[i];
  }
}

Real blockRmse(const DirichletControl& control,
               const Vector<Real>&     prm,
               const Vector<Real>&     target,
               Index                   step)
{
  Real sum = 0.0;
  for (Index i = 0; i < control.numDofs(); ++i)
  {
    const Index idx   = control.paramIndex(step, i);
    const Real  diff  = prm[idx] - target[idx];
    sum              += diff * diff;
  }
  return std::sqrt(sum / static_cast<Real>(control.numDofs()));
}

Index centerControlIndex(const MixedFESpace&     space,
                         const DirichletControl& control,
                         const TargetParams&     target)
{
  const auto  u_dof           = space.field(0);
  const Index nd              = u_dof.numComponents();
  const auto  normal          = unit(target.normal);
  Index       axial_component = 0;
  Real        axial_weight    = 0.0;
  for (Index d = 0; d < nd; ++d)
  {
    const Real weight = std::abs(normal[static_cast<std::size_t>(d)]);
    if (d == 0 || weight > axial_weight)
    {
      axial_component = d;
      axial_weight    = weight;
    }
  }

  Index best = 0;
  Real  dist = 0.0;
  for (Index i = 0; i < control.numDofs(); ++i)
  {
    const Index dof  = control.stateDof(i);
    const Index node = dof / nd;
    const Index comp = dof - nd * node;
    if (comp != axial_component)
    {
      continue;
    }

    const Real ds = sqDist(space.mesh().node(node), target.center);
    if (best == 0 || ds < dist)
    {
      best = i;
      dist = ds;
    }
  }
  return best;
}

void writeViz(const Mesh&                mesh,
              const MixedFESpace&        space,
              const DirichletControl&    control,
              const TimeStateTrajectory& target_tr,
              const TimeStateTrajectory& opt_tr,
              const Vector<Real>&        true_prm,
              const Vector<Real>&        opt_prm,
              Real                       dt,
              const VizOptions&          opts)
{
  ensureDir(opts.basename);

  TimeSeriesDataOut out;
  out.attachMesh(mesh);

  Vector<Real> ux;
  Vector<Real> uy;
  Vector<Real> uz;
  Vector<Real> p;
  Vector<Real> ux_target;
  Vector<Real> uy_target;
  Vector<Real> uz_target;
  Vector<Real> p_target;
  Vector<Real> ctr_x;
  Vector<Real> ctr_y;
  Vector<Real> ctr_z;
  Vector<Real> true_ctr_x;
  Vector<Real> true_ctr_y;
  Vector<Real> true_ctr_z;

  for (Index level = 0; level < opt_tr.numLevels(); ++level)
  {
    out.beginStep(static_cast<Real>(level) * dt);

    splitState(space, opt_tr[level], ux, uy, uz, p);
    splitState(
        space, target_tr[level], ux_target, uy_target, uz_target, p_target);

    out.addNodalVectorField("velocity_opt", ux, uy, uz);
    out.addNodalVectorField("velocity_target", ux_target, uy_target, uz_target);
    out.addNodalVectorField("velocity_error",
                            difference(ux, ux_target),
                            difference(uy, uy_target),
                            difference(uz, uz_target));
    out.addNodalScalarField("pressure_opt", p);
    out.addNodalScalarField("pressure_target", p_target);
    out.addNodalScalarField("pressure_error", difference(p, p_target));

    const Index step = level - 1;
    controlField(space, control, opt_prm, step, ctr_x, ctr_y, ctr_z);
    controlField(
        space, control, true_prm, step, true_ctr_x, true_ctr_y, true_ctr_z);
    out.addNodalVectorField("ctr_opt", ctr_x, ctr_y, ctr_z);
    out.addNodalVectorField("ctr_true", true_ctr_x, true_ctr_y, true_ctr_z);
    out.addNodalVectorField("ctr_error",
                            difference(ctr_x, true_ctr_x),
                            difference(ctr_y, true_ctr_y),
                            difference(ctr_z, true_ctr_z));
  }

  out.write(opts.basename);
}

void writeForwardViz(const Mesh&                mesh,
                     const MixedFESpace&        space,
                     const TimeStateTrajectory& tr,
                     Real                       dt,
                     const VizOptions&          opts,
                     Real                       time_offset)
{
  ensureDir(opts.basename);

  TimeSeriesDataOut out;
  out.attachMesh(mesh);

  Vector<Real> ux;
  Vector<Real> uy;
  Vector<Real> uz;
  Vector<Real> p;

  for (Index level = 0; level < tr.numLevels(); ++level)
  {
    out.beginStep(time_offset + static_cast<Real>(level) * dt);

    splitState(space, tr[level], ux, uy, uz, p);
    out.addNodalVectorField("velocity", ux, uy, uz);
    out.addNodalScalarField("pressure", p);
  }

  out.write(opts.basename);
}

} // namespace femx::navier_var
