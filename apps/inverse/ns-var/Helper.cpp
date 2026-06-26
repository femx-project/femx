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
#include <femx/linalg/BlockVectorView.hpp>

using namespace std;
using namespace femx;
using namespace femx::problem;
using namespace femx::state;
using namespace femx::assembly;
using namespace femx::fem;

namespace femx::navier_var_new
{

namespace
{

constexpr Index kQuadOrder = 2;

Point3 toPoint(const array<Real, 3>& vals)
{
  return {vals[0], vals[1], vals[2]};
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
  Vector<Index> comps;
  const auto    field = space.field(0);
  if (prm.comps.empty())
  {
    comps.reserve(field.numComponents());
    for (Index comp = 0; comp < field.numComponents(); ++comp)
    {
      comps.push_back(comp);
    }
    return comps;
  }

  comps.reserve(prm.comps.size());
  for (Index comp : prm.comps)
  {
    if (comp < 0 || comp >= field.numComponents())
    {
      throw runtime_error("observation component is out of range");
    }
    comps.push_back(comp);
  }
  return comps;
}

Vector<Point3> gridObsPoints(const ObservationParams& prm)
{
  if (!prm.grid)
  {
    throw runtime_error("grid observation requires inverse.obs.grid");
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

Vector<Point3> obsPoints(const ObservationParams& prm)
{
  if (prm.type == "grid")
  {
    return gridObsPoints(prm);
  }

  throw runtime_error(
      "Unsupported observation type: " + prm.type
      + " (expected 'grid')");
}

Vector<Point3> activeObsPoints(const MixedFESpace&      space,
                               const ObservationParams& prm)
{
  if (prm.type != "grid")
  {
    return obsPoints(prm);
  }

  const Vector<Point3> pts = gridObsPoints(prm);
  Vector<Point3>       filtered =
      TimePointInterpolator::filterPointsInside(space, 0, pts);
  if (filtered.empty())
  {
    throw runtime_error(
        "observation grid mask removed all points outside the fluid mesh");
  }
  return filtered;
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

bool matchesBoundaryFacet(const BoundarySelector&    sel,
                          const Mesh::BoundaryFacet& facet)
{
  if (!sel.name.empty())
  {
    return facet.pname == sel.name;
  }
  return facet.ptag == sel.physical;
}

Vector<ControlSpatialRegularization::Edge> controlSpatialEdges(
    const MixedFESpace&     space,
    const BoundarySelector& sel,
    const DirichletControl& ctr)
{
  const auto                     u_dof = space.field(0);
  const Index                    comps = u_dof.numComponents();
  map<pair<Index, Index>, Index> ctr_index;

  for (Index i = 0; i < ctr.numDofs(); ++i)
  {
    const Index id   = ctr.stateDof(i);
    const Index in   = id / comps;
    const Index comp = id % comps;
    if (in < 0
        || in >= space.mesh().numNodes()
        || u_dof.globalDof(in, comp) != id)
    {
      throw runtime_error(
          "controlSpatialEdges expected velocity control dofs");
    }
    ctr_index[{in, comp}] = i;
  }

  set<ControlSpatialRegularization::Edge> unique_edges;
  const auto                              add_edge =
      [&](Index in0, Index in1, Index comp)
  {
    const auto it0 = ctr_index.find({in0, comp});
    const auto it1 = ctr_index.find({in1, comp});
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
      swap(a, b);
    }
    unique_edges.insert({a, b});
  };

  for (const auto& facet : space.mesh().boundaryFacets())
  {
    if (!matchesBoundaryFacet(sel, facet)
        || facet.nids.size() < 2)
    {
      continue;
    }

    for (Index comp = 0; comp < comps; ++comp)
    {
      for (Index i = 1; i < facet.nids.size(); ++i)
      {
        add_edge(facet.nids[i - 1], facet.nids[i], comp);
      }
      if (facet.nids.size() > 2)
      {
        add_edge(facet.nids.back(), facet.nids.front(), comp);
      }
    }
  }

  Vector<ControlSpatialRegularization::Edge> edges;
  for (const auto& edge : unique_edges)
  {
    edges.push_back(edge);
  }
  return edges;
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
    throw runtime_error(
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
    throw runtime_error(
        "ParameterSliceTimeObjective base parameter gradient size mismatch");
  }

  resizeOrZero(out, total_params_);
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
    throw runtime_error(
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
    Index                  nt,
    Index                  nst,
    InverseParameterLayout lyt,
    Real                   beta)
  : nt_(nt),
    nst_(nst),
    layout_(lyt),
    beta_(beta)
{
  if (nt_ < 0 || nst_ < 0 || layout_.ntot < 0
      || layout_.init_vel_offset < 0
      || layout_.niv < 0
      || layout_.init_vel_offset + layout_.niv
             > layout_.ntot
      || beta_ < 0.0)
  {
    throw runtime_error(
        "InitialVelocityRegularization received invalid inputs");
  }
}

Index InitialVelocityRegularization::numSteps() const
{
  return nt_;
}

Index InitialVelocityRegularization::numStates() const
{
  return nst_;
}

Index InitialVelocityRegularization::numParams() const
{
  return layout_.ntot;
}

Real InitialVelocityRegularization::value(const TimeTrajectory& tr,
                                          const Vector<Real>&   prm) const
{
  (void) tr;

  const VectorView<const Real> init = layout_.initVel(prm);
  Real                         out  = 0.0;
  for (Index i = 0; i < init.size(); ++i)
  {
    const Real value  = init[i];
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
  resizeOrZero(out, nst_);
}

void InitialVelocityRegularization::paramGrad(
    const TimeTrajectory& tr,
    const Vector<Real>&   prm,
    Vector<Real>&         out) const
{
  (void) tr;

  const VectorView<const Real> init = layout_.initVel(prm);
  resizeOrZero(out, numParams());
  VectorView<Real> init_grad = layout_.initVel(out);
  for (Index i = 0; i < init.size(); ++i)
  {
    init_grad[i] = beta_ * init[i];
  }
}

ControlSpatialRegularization::ControlSpatialRegularization(
    Index        nt,
    Index        nst,
    Index        nprm,
    Index        coff,
    Index        nctr,
    Index        ncdof,
    Vector<Edge> edges,
    Real         beta)
  : nt_(nt),
    nst_(nst),
    nprm_(nprm),
    ctr_offset_(coff),
    ctr_levels_(nctr),
    ctr_dofs_(ncdof),
    edges_(std::move(edges)),
    beta_(beta)
{
  if (nt_ < 0 || nst_ < 0 || nprm_ < 0
      || ctr_offset_ < 0 || ctr_levels_ < 0
      || ctr_dofs_ < 0 || beta_ < 0.0
      || ctr_offset_ + ctr_levels_ * ctr_dofs_ > nprm_)
  {
    throw runtime_error(
        "ControlSpatialRegularization received invalid inputs");
  }
  if (beta_ > 0.0 && edges_.empty())
  {
    throw runtime_error(
        "ControlSpatialRegularization found no boundary edges");
  }
}

Index ControlSpatialRegularization::numSteps() const
{
  return nt_;
}

Index ControlSpatialRegularization::numStates() const
{
  return nst_;
}

Index ControlSpatialRegularization::numParams() const
{
  return nprm_;
}

Real ControlSpatialRegularization::value(
    const TimeTrajectory& tr,
    const Vector<Real>&   prm) const
{
  (void) tr;
  if (prm.size() != numParams())
  {
    throw runtime_error(
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
  resizeOrZero(out, nst_);
}

void ControlSpatialRegularization::paramGrad(
    const TimeTrajectory& tr,
    const Vector<Real>&   prm,
    Vector<Real>&         out) const
{
  (void) tr;
  if (prm.size() != numParams())
  {
    throw runtime_error(
        "ControlSpatialRegularization parameter size mismatch");
  }

  resizeOrZero(out, nprm_);
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

unique_ptr<TimeObservationOperator> makeObs(
    const MixedFESpace&      space,
    const ObservationParams& prm,
    Index                    steps,
    Index                    nst,
    Index                    nprm)
{
  if (nst != space.numDofs())
  {
    throw runtime_error("makeObs state size does not match FEM space");
  }

  return make_unique<TimePointInterpolator>(
      steps,
      space,
      0,
      activeObsPoints(space, prm),
      obsComponents(space, prm),
      nprm);
}

void setObsLayout(TimeObservationData&     data,
                  const MixedFESpace&      space,
                  const ObservationParams& prm)
{
  data.setLayout(
      "point", activeObsPoints(space, prm), obsComponents(space, prm));
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

Real peakBaseSpeed(const TargetParams& target)
{
  return peakSpeed(
      target.qty, "poiseuille", target.bulk_speed, 1.0, 1.5);
}

Real maxPulseSpeed(const TargetParams& target)
{
  return peakBaseSpeed(target) * (1.0 + abs(target.pulse_amplitude));
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

Vector<Real> optimizerScale(const InverseParameterLayout& lyt,
                            const OptimizerParams::Scale& scale)
{
  Vector<Real>     out(lyt.ntot);
  VectorView<Real> init = lyt.initVel(out);
  VectorView<Real> ctr  = lyt.ctr(out);
  for (Index i = 0; i < init.size(); ++i)
  {
    init[i] = scale.init_vel;
  }
  for (Index i = 0; i < ctr.size(); ++i)
  {
    ctr[i] = scale.bdry;
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

void applyInitialVelocityParamJacT(
    const Vector<Index>&          vdofs,
    const InverseParameterLayout& lyt,
    const Vector<Real>&           state_grad,
    Vector<Real>&                 out)
{
  if (vdofs.size() != lyt.niv)
  {
    throw runtime_error(
        "applyInitialVelocityParamJacT velocity id size mismatch");
  }

  resizeOrZero(out, lyt.ntot);
  VectorView<Real> init = lyt.initVel(out);
  for (Index i = 0; i < vdofs.size(); ++i)
  {
    const Index id = vdofs[i];
    if (id < 0 || id >= state_grad.size())
    {
      throw runtime_error(
          "applyInitialVelocityParamJacT velocity id is out of range");
    }
    init[i] = state_grad[id];
  }
}

void controlBounds(const MixedFESpace&     space,
                   const DirichletControl& ctr,
                   const TargetParams&     target,
                   const BoundsParams&     bnds,
                   Index                   steps,
                   Vector<Real>&           lower,
                   Vector<Real>&           upper)
{
  lower.resize(ctr.numParams(steps));
  upper.resize(ctr.numParams(steps));

  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  const auto  nrm   = unit(target.nrm);
  const Real  max_axial =
      bnds.max ? *bnds.max : bnds.max_scale * maxPulseSpeed(target);

  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < ctr.numDofs(); ++i)
    {
      const Index id   = ctr.stateDof(i);
      const Index comp = id % nd;
      const Index idx  = ctr.paramIndex(step, i);

      const Real nrm_comp = nrm[comp];
      const Real lo       = bnds.min * nrm_comp;
      const Real hi       = max_axial * nrm_comp;
      lower[idx]          = min(lo, hi);
      upper[idx]          = max(lo, hi);
    }
  }
}

void controlBounds(const MixedFESpace&     space,
                   const DirichletControl& ctr,
                   const BoundsParams&     bnds,
                   Index                   steps,
                   Vector<Real>&           lower,
                   Vector<Real>&           upper)
{
  if (!bnds.max)
  {
    throw runtime_error("controlBounds requires inverse.bounds.max");
  }

  lower.resize(ctr.numParams(steps));
  upper.resize(ctr.numParams(steps));

  const auto  u_dof = space.field(0);
  const Index nd    = u_dof.numComponents();
  const auto  nrm   = unit(bnds.nrm);

  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < ctr.numDofs(); ++i)
    {
      const Index id   = ctr.stateDof(i);
      const Index comp = id % nd;
      const Index idx  = ctr.paramIndex(step, i);

      const Real nrm_comp = nrm[comp];
      const Real lo       = bnds.min * nrm_comp;
      const Real hi       = *bnds.max * nrm_comp;
      lower[idx]          = min(lo, hi);
      upper[idx]          = max(lo, hi);
    }
  }
}

void inverseBounds(const MixedFESpace&           space,
                   const DirichletControl&       ctr,
                   const Params&                 prm,
                   const InverseParameterLayout& lyt,
                   Index                         steps,
                   Vector<Real>&                 lower,
                   Vector<Real>&                 upper)
{
  resizeOrZero(lower, lyt.ntot);
  resizeOrZero(upper, lyt.ntot);

  if (lyt.hasInitialVelocity())
  {
    constexpr Real unbounded = 1.0e30;
    const Real     lo =
        prm.inv.init_vel.lower.value_or(-unbounded);
    const Real hi =
        prm.inv.init_vel.upper.value_or(unbounded);
    VectorView<Real> init_lower = lyt.initVel(lower);
    VectorView<Real> init_upper = lyt.initVel(upper);
    for (Index i = 0; i < init_lower.size(); ++i)
    {
      init_lower[i] = lo;
      init_upper[i] = hi;
    }
  }

  Vector<Real> ctr_lower;
  Vector<Real> ctr_upper;
  if (prm.inv.bnds.max)
  {
    controlBounds(
        space,
        ctr,
        prm.inv.bnds,
        lyt.nctr,
        ctr_lower,
        ctr_upper);
  }
  else
  {
    controlBounds(space,
                  ctr,
                  controlTarget(prm),
                  prm.inv.bnds,
                  lyt.nctr,
                  ctr_lower,
                  ctr_upper);
  }
  if (ctr_lower.size() != lyt.csz
      || ctr_upper.size() != lyt.csz)
  {
    throw runtime_error(
        "inverseBounds control bound size mismatch");
  }
  VectorView<Real> ctr_lower_out = lyt.ctr(lower);
  VectorView<Real> ctr_upper_out = lyt.ctr(upper);
  for (Index i = 0; i < ctr_lower_out.size(); ++i)
  {
    ctr_lower_out[i] = ctr_lower[i];
    ctr_upper_out[i] = ctr_upper[i];
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

Objective::Objective(const Params&   prm,
                     const AppNsVar& core)
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
        0.0),
    ctr_reg(core.steps,
            core.problem.numStates(),
            core.lyt.nctr,
            core.ctr.numDofs(),
            prm.inv.reg.beta1,
            prm.inv.reg.beta2),
    reg(ctr_reg, core.problem.numParams(), core.lyt.coff),
    init_reg(core.steps,
             core.problem.numStates(),
             core.lyt,
             prm.inv.reg.beta4),
    space_reg(core.steps,
              core.problem.numStates(),
              core.problem.numParams(),
              core.lyt.coff,
              core.lyt.nctr,
              core.ctr.numDofs(),
              controlSpatialEdges(core.space, prm.inv.ctr, core.ctr),
              prm.inv.reg.beta3),
    obj(core.steps, core.problem.numStates(), core.problem.numParams())
{
  obj.add(err).add(reg).add(init_reg).add(space_reg);
}

} // namespace femx::navier_var_new
