#include "Helper.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/common/Math.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/fem/VelocityProfile.hpp>
#include <femx/linalg/BlockVectorView.hpp>

using namespace std;
using namespace femx;
using namespace femx::problem;
using namespace femx::state;
using namespace femx::assembly;
using namespace femx::fem;

namespace femx::navier_en_var
{

namespace
{

constexpr Index kQuadOrder = 2;

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

bool matchesBoundaryConfig(const BoundarySelector& sel,
                           const BCsParams&        bc)
{
  if (!sel.name.empty())
  {
    return bc.name == sel.name;
  }
  return bc.physical == sel.physical;
}

bool hasFixedVelocityValue(const BCsParams& bc)
{
  return bc.ux || bc.uy || bc.uz || bc.vel;
}

optional<Real> constantVelocityComponent(
    const BCsParams& bc,
    Index            comp)
{
  if (comp == 0 && bc.ux)
  {
    return *bc.ux;
  }
  if (comp == 1 && bc.uy)
  {
    return *bc.uy;
  }
  if (comp == 2 && bc.uz)
  {
    return *bc.uz;
  }
  return nullopt;
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
  throw runtime_error("simulation.bcs must contain a pressure boundary");
}

void addFixedValue(map<Index, Vector<Real>>& vals,
                   Index                     id,
                   Index                     step,
                   Index                     steps,
                   Real                      value)
{
  auto it = vals.find(id);
  if (it == vals.end())
  {
    it = vals.emplace(id, Vector<Real>(steps)).first;
    for (Index i = 0; i < steps; ++i)
    {
      it->second[i] = numeric_limits<Real>::quiet_NaN();
    }
  }
  else if (!isnan(it->second[step])
           && abs(it->second[step] - value) > 1.0e-12)
  {
    throw runtime_error(
        "fixed boundary has conflicting values at id "
        + to_string(id) + ", step " + to_string(step));
  }
  it->second[step] = value;
}

void addPressureGaugeValues(const MixedFESpace&       space,
                            const BCsParams&          bc,
                            Index                     steps,
                            map<Index, Vector<Real>>& vals)
{
  if (!bc.p)
  {
    return;
  }

  const Vector<Index> dofs = gaugeDofs(space, bcSelector(bc));
  for (Index step = 0; step < steps; ++step)
  {
    addFixedValue(vals, dofs[0], step, steps, *bc.p);
  }
}

Real profileVelocityValue(const MixedFESpace&     space,
                          const DirichletControl& ctr,
                          const TargetParams&     target,
                          Real                    time,
                          Index                   i);

void addProfileVelocityValues(const MixedFESpace&       space,
                              const DirichletControl&   ctr,
                              const BCsParams&          bc,
                              Index                     steps,
                              Real                      dt,
                              map<Index, Vector<Real>>& vals)
{
  if (!bc.vel)
  {
    return;
  }

  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < ctr.numDofs(); ++i)
    {
      addFixedValue(vals,
                    ctr.stateDof(i),
                    step,
                    steps,
                    profileVelocityValue(
                        space,
                        ctr,
                        *bc.vel,
                        static_cast<Real>(step + 1) * dt,
                        i));
    }
  }
}

Vector<Index> fixedvdofsForBc(const MixedFESpace& space,
                              const BCsParams&    bc)
{
  const DirichletControl bdry =
      makeVelocityControl(space, bcSelector(bc));
  if (bc.vel)
  {
    return bdry.stateDofs();
  }

  const auto    u_dof = space.field(0);
  const Index   nd    = u_dof.numComponents();
  Vector<Index> dofs;
  for (Index id : bdry.stateDofs())
  {
    const Index comp = id % nd;
    if (constantVelocityComponent(bc, comp))
    {
      dofs.push_back(id);
    }
  }
  return dofs;
}

DirichletControl makeBoundaryControl(const MixedFESpace& space,
                                     const Params&       prm,
                                     const BCsParams&    bc)
{
  const DirichletControl bdry =
      makeVelocityControl(space, bcSelector(bc));

  Vector<Index> fvdofs;
  for (const auto& fixed_bc : prm.fwd.bcs)
  {
    if (matchesBoundaryConfig(prm.inv.ctr, fixed_bc)
        || !hasFixedVelocityValue(fixed_bc))
    {
      continue;
    }
    appendUniqueExcept(
        fvdofs, fixedvdofsForBc(space, fixed_bc), {});
  }

  Vector<Index> adofs;
  adofs.reserve(bdry.numDofs());
  for (Index id : bdry.stateDofs())
  {
    if (!contains(fvdofs, id))
    {
      adofs.push_back(id);
    }
  }
  if (adofs.empty())
  {
    throw runtime_error(
        "control boundary has no active velocity dofs after fixed-boundary exclusion");
  }
  return DirichletControl(std::move(adofs));
}

void addConstantVelocityValues(const MixedFESpace&       space,
                               const DirichletControl&   ctr,
                               const BCsParams&          bc,
                               Index                     steps,
                               map<Index, Vector<Real>>& vals)
{
  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  for (Index i = 0; i < ctr.numDofs(); ++i)
  {
    const Index id    = ctr.stateDof(i);
    const Index comp  = id % nd;
    const auto  value = constantVelocityComponent(bc, comp);
    if (!value)
    {
      continue;
    }
    for (Index step = 0; step < steps; ++step)
    {
      addFixedValue(vals, id, step, steps, *value);
    }
  }
}

Real profileVelocityValue(const MixedFESpace&     space,
                          const DirichletControl& ctr,
                          const TargetParams&     target,
                          Real                    time,
                          Index                   i)
{
  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  const Index id    = ctr.stateDof(i);
  const Index in    = id / nd;
  const Index comp  = id - nd * in;

  const auto prof = poiseuilleProfile(
      target.cen, target.nrm, target.rad);
  if (comp >= 3)
  {
    return 0.0;
  }

  const Real pulse =
      sinePulseFactor(time, target.pulse_amplitude, target.per);
  const Real peak_speed =
      peakSpeed(
          target.qty, "poiseuille", target.bulk_speed, 1.0, 1.5)
      * pulse;
  return velocityComponent(
      prof, space.mesh().node(in), peak_speed, comp);
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
    throw runtime_error("observation data has no time levels");
  }

  Vector<Real> times(data.numLevels());
  for (Index row = 0; row < data.numLevels(); ++row)
  {
    times[row] =
        data.hasTimeValues()
            ? data.timeValue(row)
            : static_cast<Real>(data.timeLevel(row));
    if (!isfinite(times[row])
        || (row > 0 && times[row] <= times[row - 1]))
    {
      throw runtime_error(
          "observation times must be finite and increasing");
    }
  }
  return times;
}

Vector<Real> observationTimesOnSolveLevels(const TimeObservationData& data,
                                           Real                       dt,
                                           Index                      steps)
{
  if (steps <= 0 || dt <= 0.0 || !isfinite(dt))
  {
    throw runtime_error(
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
    throw runtime_error(
        "multiple observation times require at least two solve steps");
  }

  const Real raw_span = raw_times.back() - raw_times.front();
  const Real sim_span = final_solve_time - first_solve_time;
  if (raw_span <= 0.0 || sim_span <= 0.0)
  {
    throw runtime_error("observation time span is invalid");
  }

  for (Index row = 0; row < raw_times.size(); ++row)
  {
    const Real s = (raw_times[row] - raw_times.front()) / raw_span;
    times[row]   = first_solve_time + s * sim_span;
    if (!isfinite(times[row])
        || (row > 0 && times[row] <= times[row - 1]))
    {
      throw runtime_error(
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
    throw runtime_error("controlTimeStencils requires positive steps");
  }

  Vector<LinearInterpolation> stencils(steps);
  for (Index step = 0; step < steps; ++step)
  {
    stencils[step] = linearInterpolation(times, controlStepTime(step, dt));
  }
  return stencils;
}

void addInitialVelocityValue(map<Index, Real>& vals,
                             Index             id,
                             Real              value)
{
  const auto [it, inserted] = vals.emplace(id, value);
  if (!inserted && abs(it->second - value) > 1.0e-12)
  {
    throw runtime_error(
        "initial velocity boundary has conflicting values at id "
        + to_string(id));
  }
}

void addInitialProfileVelocityValues(const MixedFESpace&     space,
                                     const DirichletControl& ctr,
                                     const TargetParams&     target,
                                     Real                    time,
                                     map<Index, Real>&       vals)
{
  for (Index i = 0; i < ctr.numDofs(); ++i)
  {
    addInitialVelocityValue(vals,
                            ctr.stateDof(i),
                            profileVelocityValue(
                                space, ctr, target, time, i));
  }
}

void addInitialConstantVelocityValues(const MixedFESpace&     space,
                                      const DirichletControl& ctr,
                                      const BCsParams&        bc,
                                      bool                    fill_missing,
                                      map<Index, Real>&       vals)
{
  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  for (Index i = 0; i < ctr.numDofs(); ++i)
  {
    const Index id    = ctr.stateDof(i);
    const Index comp  = id % nd;
    const auto  value = constantVelocityComponent(bc, comp);
    if (!value && !fill_missing)
    {
      continue;
    }
    addInitialVelocityValue(vals, id, value.value_or(0.0));
  }
}

void addInitialBoundaryVelocityValues(const MixedFESpace&     space,
                                      const DirichletControl& ctr,
                                      const BCsParams&        bc,
                                      bool                    fill_missing,
                                      map<Index, Real>&       vals)
{
  if (bc.vel)
  {
    addInitialProfileVelocityValues(
        space, ctr, *bc.vel, 0.0, vals);
    return;
  }
  addInitialConstantVelocityValues(
      space, ctr, bc, fill_missing, vals);
}

FixedDofValues toFixedDofValues(
    const map<Index, Vector<Real>>& vals,
    Index                           steps)
{
  FixedDofValues out;
  for (const auto& entry : vals)
  {
    out.dofs.push_back(entry.first);
  }

  out.vals.resize(steps * out.dofs.size());
  BlockVectorView<Real> values(out.vals.data(), steps, out.dofs.size());
  Index                 i = 0;
  for (const auto& entry : vals)
  {
    for (Index step = 0; step < steps; ++step)
    {
      if (isnan(entry.second[step]))
      {
        throw runtime_error(
            "fixed boundary value was not assigned for every time step");
      }
      values(step, i) = entry.second[step];
    }
    ++i;
  }
  return out;
}

} // namespace

void initialStateFromParams(
    const Vector<Index>&          vdofs,
    const InverseParameterLayout& lyt,
    const Vector<Real>&           base_state,
    const Vector<Real>&           prm,
    Vector<Real>&                 out);

InitialVelocityStateSolver::InitialVelocityStateSolver(
    TimeLinearStateSolver& solver,
    Vector<Index>          vdofs,
    InverseParameterLayout lyt,
    Vector<Real>           x0)
  : solver_(solver),
    vdofs_(std::move(vdofs)),
    layout_(lyt),
    x0_(std::move(x0))
{
  if (!layout_.hasInitialVelocity())
  {
    throw runtime_error(
        "InitialVelocityStateSolver requires initial velocity parameters");
  }
  if (solver_.numParams() != layout_.ntot)
  {
    throw runtime_error(
        "InitialVelocityStateSolver parameter size mismatch");
  }
  if (vdofs_.size() != layout_.niv)
  {
    throw runtime_error(
        "InitialVelocityStateSolver velocity id size mismatch");
  }
  if (x0_.empty())
  {
    x0_.resize(solver_.numStates());
  }
  else if (x0_.size() != solver_.numStates())
  {
    throw runtime_error(
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
  initialStateFromParams(vdofs_, layout_, x0_, prm, initial_state_);
  solver_.setInitialState(initial_state_);
  solver_.solve(prm, tr);
}

Vector<Index> initialvdofs(const MixedFESpace& space)
{
  const auto  u_dof = space.field(0);
  const Index nodes = space.mesh().numNodes();
  const Index comps = u_dof.numComponents();

  Vector<Index> dofs;
  dofs.reserve(nodes * comps);
  for (Index in = 0; in < nodes; ++in)
  {
    for (Index comp = 0; comp < comps; ++comp)
    {
      dofs.push_back(u_dof.globalDof(in, comp));
    }
  }
  return dofs;
}

Vector<Index> initialvdofs(const MixedFESpace&     space,
                           const Params&           prm,
                           const DirichletControl& ctr)
{
  Vector<Index> constrained = ctr.stateDofs();
  for (const auto& bc : prm.fwd.bcs)
  {
    if (matchesBoundaryConfig(prm.inv.ctr, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }
    appendUniqueExcept(
        constrained, fixedvdofsForBc(space, bc), ctr.stateDofs());
  }

  const Vector<Index> all = initialvdofs(space);
  Vector<Index>       out;
  out.reserve(all.size());
  for (Index id : all)
  {
    if (!contains(constrained, id))
    {
      out.push_back(id);
    }
  }
  return out;
}

Vector<Real> initialVelocityBoundaryState(
    const MixedFESpace&     space,
    const Params&           prm,
    const DirichletControl& ctr)
{
  map<Index, Real> vals;

  addInitialBoundaryVelocityValues(
      space, ctr, controlBoundary(prm), true, vals);

  for (const auto& bc : prm.fwd.bcs)
  {
    if (matchesBoundaryConfig(prm.inv.ctr, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }

    const Vector<Index> fdofs = fixedvdofsForBc(space, bc);
    Vector<Index>       adofs;
    for (Index id : fdofs)
    {
      if (!contains(ctr.stateDofs(), id))
      {
        adofs.push_back(id);
      }
    }
    if (adofs.empty())
    {
      continue;
    }
    const DirichletControl active_fixed(adofs);
    addInitialBoundaryVelocityValues(
        space, active_fixed, bc, false, vals);
  }

  Vector<Real> out(space.numDofs());
  out.setZero();
  for (const auto& entry : vals)
  {
    if (entry.first < 0 || entry.first >= out.size())
    {
      throw runtime_error(
          "initial velocity boundary id is out of range");
    }
    out[entry.first] = entry.second;
  }
  return out;
}

InverseParameterLayout inverseParameterLayout(
    const MixedFESpace&          space,
    const DirichletControl&      ctr,
    const InitialVelocityParams& init_vel,
    Index                        steps,
    Index                        niv,
    Index                        nctr)
{
  (void) space;
  if (steps <= 0 || niv < 0 || nctr <= 0)
  {
    throw runtime_error(
        "inverseParameterLayout received invalid sizes");
  }

  InverseParameterLayout lyt;
  if (init_vel.enabled)
  {
    lyt.init_vel_offset = 0;
    lyt.niv             = niv;
  }
  lyt.coff = lyt.init_vel_offset
             + lyt.niv;
  lyt.nctr = nctr;
  lyt.csz  = nctr * ctr.numDofs();
  lyt.ntot = lyt.coff + lyt.csz;
  return lyt;
}

FixedDofValues fixedDofValues(const MixedFESpace&     space,
                              const Params&           prm,
                              const DirichletControl& ctr,
                              Index                   steps,
                              Real                    dt)
{
  if (steps <= 0)
  {
    throw runtime_error("fixedDofValues requires positive steps");
  }

  map<Index, Vector<Real>> vals;
  addPressureGaugeValues(
      space, pressureGaugeBoundary(prm), steps, vals);

  for (const auto& bc : prm.fwd.bcs)
  {
    if (matchesBoundaryConfig(prm.inv.ctr, bc)
        || !hasFixedVelocityValue(bc))
    {
      continue;
    }

    const Vector<Index> fdofs = fixedvdofsForBc(space, bc);
    Vector<Index>       adofs;
    for (Index id : fdofs)
    {
      if (!contains(ctr.stateDofs(), id))
      {
        adofs.push_back(id);
      }
    }
    if (adofs.empty())
    {
      continue;
    }
    const DirichletControl active_fixed(adofs);

    addProfileVelocityValues(space,
                             active_fixed,
                             bc,
                             steps,
                             dt,
                             vals);
    addConstantVelocityValues(
        space, active_fixed, bc, steps, vals);
  }

  return toFixedDofValues(vals, steps);
}

Index observationVectorSize(const TimeObservationData& data)
{
  return data.numLevels() * data.numObservations();
}

Vector<Real> flattenObservations(const TimeObservationData& data)
{
  Vector<Real> out(observationVectorSize(data));
  Index        offset = 0;
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    const Vector<Real> vals = data[level];
    for (Index i = 0; i < vals.size(); ++i)
    {
      out[offset++] = vals[i];
    }
  }
  return out;
}

LinearInterpolation observationInterpolation(const TimeObservationData& data,
                                             Index                      row,
                                             Index                      steps,
                                             Real                       dt,
                                             Real                       time_offset)
{
  if (row < 0 || row >= data.numLevels())
  {
    throw runtime_error("observationInterpolation row is out of range");
  }
  if (data.hasTimeValues())
  {
    if (dt <= 0.0 || !isfinite(dt))
    {
      throw runtime_error("observationInterpolation requires positive dt");
    }

    const Real scaled = (data.timeValue(row) + time_offset) / dt;
    const Real tol =
        max<Real>(1.0e-10,
                  1.0e-8 * max<Real>(1.0, abs(scaled)));
    if (scaled < -tol || scaled > static_cast<Real>(steps) + tol)
    {
      throw runtime_error("observation time is out of range");
    }

    const Real clamped =
        min<Real>(max<Real>(scaled, 0.0), static_cast<Real>(steps));
    const Index nearest = static_cast<Index>(llround(clamped));
    if (abs(clamped - static_cast<Real>(nearest)) <= tol)
    {
      return {nearest, nearest, 0.0};
    }

    const Index lower = static_cast<Index>(floor(clamped));
    return {lower, lower + 1, clamped - static_cast<Real>(lower)};
  }

  Index level = data.hasTimeLevels() ? data.timeLevel(row) : row;
  if (time_offset != 0.0)
  {
    if (dt <= 0.0 || !isfinite(dt))
    {
      throw runtime_error(
          "observationInterpolation requires positive dt for time offset");
    }
    const Real  offset       = time_offset / dt;
    const Index level_offset = static_cast<Index>(llround(offset));
    if (abs(offset - static_cast<Real>(level_offset)) > 1.0e-8)
    {
      throw runtime_error(
          "observationInterpolation time offset must align to a time step");
    }
    level += level_offset;
  }
  if (level < 0 || level > steps)
  {
    throw runtime_error("observation time level is out of range");
  }
  return {level, level, 0.0};
}

void observeInterpolated(const TimeObservationOperator& obs,
                         const LinearInterpolation&     interp,
                         const TimeTrajectory&          tr,
                         const Vector<Real>&            prm,
                         Vector<Real>&                  out)
{
  obs.observe(interp.lower, tr[interp.lower], prm, out);
  if (!interp.hasUpper())
  {
    return;
  }

  Vector<Real> upper;
  obs.observe(interp.upper, tr[interp.upper], prm, upper);
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = interp.lowerWeight() * out[i] + interp.upperWeight() * upper[i];
  }
}

TimeObservationData sampleObservationData(
    const TimeObservationOperator& obs,
    const TimeObservationData&     layout,
    const TimeTrajectory&          tr,
    const Vector<Real>&            prm,
    Real                           dt,
    Real                           time_offset)
{
  if (tr.numSteps() != obs.numSteps() || tr.numStates() != obs.numStates()
      || prm.size() != obs.numParams()
      || layout.numObservations() != obs.numObservations())
  {
    throw runtime_error("sampleObservationData received inconsistent inputs");
  }

  TimeObservationData out = layout;
  for (Index row = 0; row < out.numLevels(); ++row)
  {
    const LinearInterpolation interp =
        observationInterpolation(out, row, obs.numSteps(), dt, time_offset);
    Vector<Real> vals = out[row];
    observeInterpolated(obs, interp, tr, prm, vals);
  }
  return out;
}

Vector<Real> wError(Index nt,
                    Real  wt,
                    bool  include_initial)
{
  Vector<Real> wts(nt + 1);
  wts[0] = include_initial ? wt : 0.0;
  for (Index level = 1; level <= nt; ++level)
  {
    wts[level] = wt;
  }
  return wts;
}

Vector<Real> makeTrueParams(const MixedFESpace&     space,
                            const DirichletControl& ctr,
                            const TargetParams&     target,
                            const Vector<Real>&     times)
{
  Vector<Real>          prm(times.size() * ctr.numDofs());
  BlockVectorView<Real> levels(prm.data(), times.size(), ctr.numDofs());

  for (Index level = 0; level < times.size(); ++level)
  {
    VectorView<Real> values = levels.block(level);
    for (Index i = 0; i < ctr.numDofs(); ++i)
    {
      values[i] =
          profileVelocityValue(space, ctr, target, times[level], i);
    }
  }
  return prm;
}

optional<Real> componentValue(const BCsParams& bc, Index comp)
{
  if (comp == 0 && bc.ux)
  {
    return *bc.ux;
  }
  if (comp == 1 && bc.uy)
  {
    return *bc.uy;
  }
  if (comp == 2 && bc.uz)
  {
    return *bc.uz;
  }
  return nullopt;
}

Vector<Real> initialControlParams(const MixedFESpace&     space,
                                  const DirichletControl& ctr,
                                  const BCsParams&        bc,
                                  const Vector<Real>&     times)
{
  if (times.empty())
  {
    throw runtime_error("initialControlParams requires control times");
  }

  if (bc.vel)
  {
    return makeTrueParams(space, ctr, *bc.vel, times);
  }

  Vector<Real> prm(times.size() * ctr.numDofs());
  prm.setZero();
  BlockVectorView<Real> levels(prm.data(), times.size(), ctr.numDofs());

  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  for (Index level = 0; level < times.size(); ++level)
  {
    VectorView<Real> values = levels.block(level);
    for (Index i = 0; i < ctr.numDofs(); ++i)
    {
      const Index id   = ctr.stateDof(i);
      const Index comp = id % nd;
      if (const auto value = componentValue(bc, comp))
      {
        values[i] = *value;
      }
    }
  }
  return prm;
}

Vector<Real> initialInverseParams(const MixedFESpace&           space,
                                  const DirichletControl&       ctr,
                                  const Params&                 prm,
                                  const InverseParameterLayout& lyt,
                                  Index                         steps,
                                  Real                          dt,
                                  const Vector<Real>&           ctr_times)
{
  (void) steps;
  (void) dt;
  Vector<Real> out(lyt.ntot);
  out.setZero();

  const Vector<Real> ctr_prm =
      initialControlParams(
          space,
          ctr,
          controlBoundary(prm),
          ctr_times);
  if (ctr_prm.size() != lyt.csz)
  {
    throw runtime_error(
        "initialInverseParams control parameter size mismatch");
  }
  VectorView<Real> ctr_out = lyt.ctr(out);
  for (Index i = 0; i < ctr_prm.size(); ++i)
  {
    ctr_out[i] = ctr_prm[i];
  }
  return out;
}

void setControlParams(const InverseParameterLayout& lyt,
                      const Vector<Real>&           ctr_prm,
                      Vector<Real>&                 prm)
{
  if (prm.size() != lyt.ntot
      || ctr_prm.size() != lyt.csz)
  {
    throw runtime_error("setControlParams size mismatch");
  }

  VectorView<Real> ctr_dst = lyt.ctr(prm);
  for (Index i = 0; i < ctr_dst.size(); ++i)
  {
    ctr_dst[i] = ctr_prm[i];
  }
}

Vector<Real> controlParams(const InverseParameterLayout& lyt,
                           const Vector<Real>&           prm)
{
  const VectorView<const Real> ctr_src = lyt.ctr(prm);
  Vector<Real>                 out(ctr_src.size());
  for (Index i = 0; i < ctr_src.size(); ++i)
  {
    out[i] = ctr_src[i];
  }
  return out;
}

void setInitialVelocityParams(const Vector<Index>&          vdofs,
                              const InverseParameterLayout& lyt,
                              const Vector<Real>&           state,
                              Vector<Real>&                 prm)
{
  if (prm.size() != lyt.ntot
      || vdofs.size() != lyt.niv)
  {
    throw runtime_error("setInitialVelocityParams size mismatch");
  }

  VectorView<Real> init = lyt.initVel(prm);
  for (Index i = 0; i < vdofs.size(); ++i)
  {
    const Index id = vdofs[i];
    if (id < 0 || id >= state.size())
    {
      throw runtime_error(
          "setInitialVelocityParams velocity id is out of range");
    }
    init[i] = state[id];
  }
}

void initializeOptGuess(const MixedFESpace&           space,
                        const DirichletControl&       ctr,
                        const Params&                 prm,
                        const InverseParameterLayout& lyt,
                        const Vector<Index>&          vdofs,
                        TimeLinearStateSolver&        state_solver,
                        const Vector<Real>&           ctr_times,
                        Vector<Real>&                 prm_init,
                        Vector<Real>*                 x0)
{
  if (prm_init.size() != lyt.ntot)
  {
    throw runtime_error(
        "initializeOptGuess parameter size mismatch");
  }

  const Vector<Real> ctr_prm = initialControlParams(space, ctr, controlBoundary(prm), ctr_times);
  setControlParams(lyt, ctr_prm, prm_init);

  Vector<Real> guess_prm(lyt.ntot);
  guess_prm.setZero();
  setControlParams(lyt, ctr_prm, guess_prm);

  TimeTrajectory tr;
  state_solver.solve(guess_prm, tr);

  const Vector<Real>& final_state = tr[tr.numSteps()];
  if (x0 != nullptr)
  {
    *x0 = final_state;
  }
  setInitialVelocityParams(vdofs, lyt, final_state, prm_init);
}

void initialStateFromParams(const Vector<Index>&          vdofs,
                            const InverseParameterLayout& lyt,
                            Index                         nst,
                            const Vector<Real>&           prm,
                            Vector<Real>&                 out)
{
  Vector<Real> base_state(nst);
  base_state.setZero();
  initialStateFromParams(vdofs, lyt, base_state, prm, out);
}

void initialStateFromParams(const Vector<Index>&          vdofs,
                            const InverseParameterLayout& lyt,
                            const Vector<Real>&           base_state,
                            const Vector<Real>&           prm,
                            Vector<Real>&                 out)
{
  if (prm.size() != lyt.ntot)
  {
    throw runtime_error(
        "initialStateFromParams parameter size mismatch");
  }
  if (vdofs.size() != lyt.niv)
  {
    throw runtime_error(
        "initialStateFromParams velocity id size mismatch");
  }

  if (base_state.empty())
  {
    throw runtime_error(
        "initialStateFromParams base state is empty");
  }
  out                               = base_state;
  const VectorView<const Real> init = lyt.initVel(prm);
  for (Index i = 0; i < vdofs.size(); ++i)
  {
    const Index id = vdofs[i];
    if (id < 0 || id >= out.size())
    {
      throw runtime_error(
          "initialStateFromParams velocity id is out of range");
    }
    out[id] = init[i];
  }
}

void checkInverseRunParams(const Params& prm)
{
  if (prm.fwd.time.steps <= 0 || prm.fwd.time.dt <= 0.0)
  {
    throw runtime_error("time steps and dt must be positive");
  }
  if (prm.inv.obs.file.empty())
  {
    throw runtime_error("inverse.obs.file is required");
  }
  if (prm.inv.ens.perturbations_file.empty())
  {
    throw runtime_error(
        "inverse.ensemble.perturbations_file is required");
  }
}

Vector<Real> readEnsembleVector(const string& path)
{
  ifstream in(path);
  if (!in)
  {
    throw runtime_error("Failed to open ensemble vector file: " + path);
  }

  Vector<Real> out;
  Real         value = 0.0;
  while (in >> value)
  {
    out.push_back(value);
  }
  if (!in.eof())
  {
    throw runtime_error("Failed to read ensemble vector file: " + path);
  }
  return out;
}

Vector<Real> readSizedEnsembleVector(const string& path, Index size)
{
  ifstream in(path);
  if (!in)
  {
    throw runtime_error("Failed to open ensemble vector file: " + path);
  }

  Index rows = 0;
  if (!(in >> rows))
  {
    throw runtime_error(
        "Ensemble vector file must begin with a value count");
  }
  if (rows != size)
  {
    throw runtime_error(
        "Ensemble vector dimensions do not match expected size");
  }

  Vector<Real> out(rows);
  for (Index i = 0; i < rows; ++i)
  {
    if (!(in >> out[i]))
    {
      throw runtime_error(
          "Ensemble vector file ended before all values were read");
    }
  }

  Real extra = 0.0;
  if (in >> extra)
  {
    throw runtime_error("Ensemble vector file contains extra values");
  }
  return out;
}

DenseMatrix readEnsemblePerturbations(const string& path, Index nprm)
{
  ifstream in(path);
  if (!in)
  {
    throw runtime_error(
        "Failed to open ensemble perturbations file: " + path);
  }

  Index rows = 0;
  Index cols = 0;
  if (!(in >> rows >> cols))
  {
    throw runtime_error(
        "Ensemble perturbations file must begin with row and column counts");
  }
  if (rows != nprm || cols <= 0)
  {
    throw runtime_error(
        "Ensemble perturbations dimensions do not match parameter layout");
  }

  DenseMatrix out(rows, cols);
  for (Index i = 0; i < rows; ++i)
  {
    for (Index j = 0; j < cols; ++j)
    {
      if (!(in >> out(i, j)))
      {
        throw runtime_error(
            "Ensemble perturbations file ended before all values were read");
      }
    }
  }

  Real extra = 0.0;
  if (in >> extra)
  {
    throw runtime_error(
        "Ensemble perturbations file contains extra values");
  }
  return out;
}

state::EnsembleBasis ensembleBasis(const EnsembleParams& prm,
                                   const Vector<Real>&   fallback_mean,
                                   Index                 nprm)
{
  if (fallback_mean.size() != nprm)
  {
    throw runtime_error("Ensemble fallback mean size mismatch");
  }

  Vector<Real> mean =
      prm.mean_file.empty() ? fallback_mean
                            : readEnsembleVector(prm.mean_file);
  if (mean.size() != nprm)
  {
    throw runtime_error("Ensemble mean size mismatch");
  }

  DenseMatrix perturbations =
      readEnsemblePerturbations(prm.perturbations_file, nprm);
  return state::EnsembleBasis(std::move(mean), std::move(perturbations));
}

state::EnsembleBasis observationEnsembleBasis(const EnsembleParams& prm,
                                              Index                 nobs,
                                              Index                 ncoef)
{
  if (nobs <= 0 || ncoef <= 0)
  {
    throw runtime_error("Observation ensemble dimensions must be positive");
  }
  if (prm.obs_mean_file.empty())
  {
    throw runtime_error("Observation ensemble mean file is required");
  }
  if (prm.obs_perturbations_file.empty())
  {
    throw runtime_error("Observation ensemble perturbations file is required");
  }

  Vector<Real> mean = readSizedEnsembleVector(prm.obs_mean_file, nobs);
  DenseMatrix  perturbations =
      readEnsemblePerturbations(prm.obs_perturbations_file, nobs);
  if (perturbations.cols() != ncoef)
  {
    throw runtime_error(
        "Observation ensemble coefficient dimension mismatch");
  }
  return state::EnsembleBasis(std::move(mean), std::move(perturbations));
}

AppNsEnVar::AppNsEnVar(const Params& prm)
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
        Vector<DofLayout>{DofLayout(space), DofLayout(space)},
        DofLayout(space),
        ns),
    bc(controlBoundary(prm)),
    ctr(makeBoundaryControl(space, prm, bc)),
    init_vdofs(
        prm.inv.init_vel.enabled ? initialvdofs(space, prm, ctr)
                                 : Vector<Index>{}),
    obs_data(readTimeObsData(prm.inv.obs.file)),
    ctr_times(controlKnotTimes(obs_data, dt, steps)),
    ctr_time_stencils(controlTimeStencils(steps, dt, ctr_times)),
    lyt(inverseParameterLayout(space,
                               ctr,
                               prm.inv.init_vel,
                               steps,
                               init_vdofs.size(),
                               ctr_times.size())),
    fixed(fixedDofValues(space, prm, ctr, steps, dt)),
    problem(fem,
            ctr,
            fixed.dofs,
            lyt.coff,
            lyt.ntot,
            fixed.vals,
            ctr_time_stencils),
    x0(initialVelocityBoundaryState(space, prm, ctr)),
    pettern(SparsityPatternBuilder::build(space)),
    prm0(initialInverseParams(
        space, ctr, prm, lyt, steps, dt, ctr_times))
{
  if (prm0.size() != problem.numParams())
  {
    throw runtime_error("initial inverse parameter size mismatch");
  }
}

Objective::Objective(const Params&     prm,
                     const AppNsEnVar& core)
  : data(observationDataOnSolveLevels(core.obs_data, core.dt, core.steps)),
    op(core.steps,
       core.space,
       0,
       data.pts(),
       data.comps(),
       core.problem.numParams()),
    err(op,
        data,
        wError(core.steps, prm.inv.alpha, false),
        core.dt,
        0.0)
{
}

} // namespace femx::navier_en_var
