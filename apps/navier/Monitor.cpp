#include "Monitor.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <femx/common/Math.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/model/navier/StateFields.hpp>
#include <femx/runtime/Output.hpp>

namespace femx::apps::navier
{
using namespace io;

namespace
{

Real elemMinEdge(const fem::Mesh& mesh, Index ie)
{
  Real h = std::numeric_limits<Real>::infinity();
  for (Index in = 0; in < mesh.elemNumNodes(ie); ++in)
  {
    for (Index jn = in + 1; jn < mesh.elemNumNodes(ie); ++jn)
    {
      h = std::min(h, distance(mesh.elemNode(ie, in), mesh.elemNode(ie, jn)));
    }
  }
  return std::isfinite(h) ? h : 0.0;
}

std::string stepLogLine(Index step,
                        Real  time,
                        Real  max_cfl,
                        bool  show_velocity_change,
                        Real  vel_change,
                        Real  assm_sec,
                        Real  solve_sec)
{
  std::ostringstream line;
  line << "step " << std::setw(7) << step << ", t = " << std::setw(11) << time
       << ", max CFL = " << std::setw(11) << max_cfl;
  if (show_velocity_change)
  {
    line << ", rel du = " << std::setw(11) << vel_change;
  }
  line << ", assembly = " << std::setw(11) << assm_sec << " s"
       << ", solve = " << std::setw(11) << solve_sec << " s";
  return line.str();
}

void writeLine(const std::string& line,
               std::ostream*      terminal,
               std::ostream*      log_out)
{
  if (terminal != nullptr)
  {
    *terminal << line << '\n';
  }
  if (log_out != nullptr)
  {
    *log_out << line << '\n';
    log_out->flush();
  }
}

#ifndef FEMX_HAS_HDF5
void packVelocity(const HostVector<Real>& ux,
                  const HostVector<Real>& uy,
                  const HostVector<Real>& uz,
                  HostVector<Real>&       vel)
{
  const Index num_nodes = ux.size();
  if (uy.size() != num_nodes
      || uz.size() != num_nodes
      || vel.size() != 3 * num_nodes)
  {
    throw std::runtime_error("Velocity field output received inconsistent sizes");
  }

  for (Index in = 0; in < num_nodes; ++in)
  {
    vel[3 * in]     = ux[in];
    vel[3 * in + 1] = uy[in];
    vel[3 * in + 2] = uz[in];
  }
}

std::string stepVtuFile(const std::string& dir,
                        Index              level)
{
  std::ostringstream file;
  file << dir << "/fields_"
       << std::setw(6) << std::setfill('0') << level
       << ".vtu";
  return file.str();
}
#endif

} // namespace

struct Monitor::FieldOutput
{
  explicit FieldOutput(const fem::Mesh& mesh)
    : vel(3 * mesh.numNodes()),
      ux(mesh.numNodes()),
      uy(mesh.numNodes()),
      uz(mesh.numNodes()),
      p(mesh.numNodes())
  {
    vel_out.attachMesh(mesh);
    pre_out.attachMesh(mesh);
  }

  TimeSeriesDataOut vel_out;
  TimeSeriesDataOut pre_out;
  VtuWriter         vtu_out;
  HostVector<Real>  vel;
  HostVector<Real>  ux;
  HostVector<Real>  uy;
  HostVector<Real>  uz;
  HostVector<Real>  p;
};

Monitor::Monitor(const fem::MixedFESpace& space,
                 Real                     dt,
                 Index                    steps)
  : space_(&space),
    dt_(dt),
    num_steps_(steps)
{
}

Monitor::~Monitor() = default;

void Monitor::setFieldOutput(std::string dir,
                             Index       interval)
{
  field_dir_      = std::move(dir);
  field_interval_ = interval;
}

void Monitor::setDetailedLog(std::ostream* terminal,
                             std::ostream* log_out,
                             bool          show_velocity_change)
{
  log_terminal_    = terminal;
  log_out_         = log_out;
  show_vel_change_ = show_velocity_change;
}

void Monitor::setConvergence(ConvergenceConfig prm)
{
  convergence_ = prm;
}

const SolveResult& Monitor::result() const
{
  return res_;
}

void Monitor::start(Index num_steps,
                    Index num_states)
{
  (void) num_states;

  if (num_steps_ <= 0)
  {
    num_steps_ = num_steps;
  }
  res_             = SolveResult{};
  res_.vel_change  = std::numeric_limits<Real>::quiet_NaN();
  last_field_step_ = 0;

  if (fieldOutputEnabled())
  {
    runtime::ensureDirectory(field_dir_);
    field_out_ = std::make_unique<FieldOutput>(space_->mesh());
  }
}

void Monitor::observe(Index                   level,
                      const HostVector<Real>& state)
{
  if (level > 0)
  {
    res_.final_step  = level;
    res_.final_time  = static_cast<Real>(level) * dt_;
    res_.final_state = state;
    if (fieldOutputEnabled()
        && shouldWriteOutput(level, num_steps_, field_interval_))
    {
      writeFieldOutput(level, state, res_.final_time);
    }
  }
}

bool Monitor::observeStep(const state::TimeStepStateContext& ctx)
{
  res_.final_step  = ctx.level;
  res_.final_time  = static_cast<Real>(ctx.level) * dt_;
  res_.final_state = ctx.curr;

  const bool need_velocity_change =
      convergence_.enabled || show_vel_change_;
  if (need_velocity_change)
  {
    res_.vel_change = velocityRelativeChange(*space_, ctx.prev, ctx.curr);
  }

  res_.converged =
      convergence_.enabled
      && ctx.level >= convergence_.min_steps
      && res_.vel_change < convergence_.vel_rel_tol;

  if (fieldOutputEnabled() && shouldWriteOutput(ctx.level, num_steps_, field_interval_))
  {
    writeFieldOutput(ctx.level, ctx.curr, res_.final_time);
  }

  if (detailedLogEnabled()
      && shouldWriteDetailedLog(ctx.level, ctx.total_steps))
  {
    const Real max_cfl = maxVelocityCfl(*space_, ctx.prev, dt_);
    if (!std::isfinite(max_cfl))
    {
      throw std::runtime_error("Stopping as CFL became invalid");
    }
    writeDetailedStepLog(ctx.level,
                         res_.final_time,
                         max_cfl,
                         res_.vel_change,
                         ctx.assm_sec,
                         ctx.lin_solve_sec);
  }

  return res_.converged;
}

void Monitor::stop()
{
  writeFinalFieldOutput();
}

bool Monitor::fieldOutputEnabled() const
{
  return !field_dir_.empty() && field_interval_ > 0;
}

bool Monitor::detailedLogEnabled() const
{
  return log_terminal_ != nullptr || log_out_ != nullptr;
}

bool Monitor::shouldWriteDetailedLog(Index step,
                                     Index total) const
{
  if (fieldOutputEnabled())
  {
    return shouldWriteOutput(step, total, field_interval_);
  }
  return true;
}

void Monitor::writeFieldOutput(Index                   level,
                               const HostVector<Real>& state,
                               Real                    time)
{
  if (field_out_ == nullptr)
  {
    return;
  }

  model::navier::splitStateFields(
      HostVectorView<const Real>(state.data(), state.size()),
      *space_,
      field_out_->ux,
      field_out_->uy,
      field_out_->uz,
      field_out_->p);

#ifdef FEMX_HAS_HDF5
  field_out_->vel_out.beginStep(time);
  field_out_->vel_out.addNodalVectorField("velocity",
                                          field_out_->ux,
                                          field_out_->uy,
                                          field_out_->uz);

  field_out_->pre_out.beginStep(time);
  field_out_->pre_out.addNodalScalarField("pressure",
                                          field_out_->p);

  field_out_->vel_out.write(field_dir_ + "/velocity");
  field_out_->pre_out.write(field_dir_ + "/pressure");
#else
  (void) time;
  packVelocity(field_out_->ux,
               field_out_->uy,
               field_out_->uz,
               field_out_->vel);
  field_out_->vtu_out.writePointData(
      stepVtuFile(field_dir_, level),
      space_->mesh(),
      HostVector<VtuWriter::PointField>{
          {"velocity", 3, &field_out_->vel},
          {"pressure", 1, &field_out_->p}});
#endif
  last_field_step_ = level;
}

void Monitor::writeFinalFieldOutput()
{
  if (!fieldOutputEnabled()
      || res_.final_step <= 0
      || last_field_step_ == res_.final_step)
  {
    return;
  }
  writeFieldOutput(res_.final_step,
                   res_.final_state,
                   res_.final_time);
}

void Monitor::writeDetailedStepLog(Index step,
                                   Real  time,
                                   Real  max_cfl,
                                   Real  vel_change,
                                   Real  assm_sec,
                                   Real  solve_sec)
{
  writeLine(stepLogLine(step,
                        time,
                        max_cfl,
                        show_vel_change_,
                        vel_change,
                        assm_sec,
                        solve_sec),
            log_terminal_,
            log_out_);
}

Real velocityRelativeChange(const fem::MixedFESpace& space,
                            const HostVector<Real>&  prev,
                            const HostVector<Real>&  curr)
{
  if (prev.size() != curr.size() || prev.size() != space.numDofs())
  {
    throw std::runtime_error("velocity convergence received incompatible states");
  }

  const auto  vel       = space.field(0);
  const Index num_nodes = vel.space().mesh().numNodes();
  const Index comps     = vel.numComponents();

  Real diff2 = 0.0;
  Real ref2  = 0.0;
  for (Index in = 0; in < num_nodes; ++in)
  {
    for (Index ic = 0; ic < comps; ++ic)
    {
      const Index dof   = vel.globalDof(in, ic);
      const Real  diff  = curr[dof] - prev[dof];
      diff2            += diff * diff;
      ref2             += prev[dof] * prev[dof];
    }
  }

  if (diff2 <= 0.0)
  {
    return 0.0;
  }
  if (ref2 <= 0.0)
  {
    return std::numeric_limits<Real>::infinity();
  }
  return std::sqrt(diff2 / ref2);
}

Real maxVelocityCfl(const fem::MixedFESpace& space,
                    const HostVector<Real>&  state,
                    Real                     dt)
{
  if (state.size() != space.numDofs())
  {
    throw std::runtime_error("CFL calculation received incompatible state size");
  }

  const auto  vel     = space.field(0);
  const Index comps   = vel.numComponents();
  Real        max_cfl = 0.0;

  for (Index ie = 0; ie < space.mesh().numElems(); ++ie)
  {
    const Real h = elemMinEdge(space.mesh(), ie);
    if (h <= 0.0)
    {
      continue;
    }

    for (Index in = 0; in < space.mesh().elemNumNodes(ie); ++in)
    {
      const Index node = space.mesh().elemNodeId(ie, in);
      Real        vel2 = 0.0;
      for (Index ic = 0; ic < comps; ++ic)
      {
        const Real val  = state[vel.globalDof(node, ic)];
        vel2           += val * val;
      }
      max_cfl = std::max(max_cfl, std::sqrt(vel2) * dt / h);
    }
  }

  return max_cfl;
}

bool shouldWriteOutput(Index step,
                       Index total_steps,
                       Index interval)
{
  return interval > 0 && (step % interval == 0 || step == total_steps);
}

} // namespace femx::apps::navier
