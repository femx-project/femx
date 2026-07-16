#include "ForwardSolveMonitor.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "Helper.hpp"
#include <femx/common/Math.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/runtime/Output.hpp>

namespace femx::model::ns
{
using namespace io;

namespace
{

Real elemMinEdge(const fem::Element& elem)
{
  Real h = std::numeric_limits<Real>::infinity();
  for (Index i = 0; i < elem.numNodes(); ++i)
  {
    for (Index j = i + 1; j < elem.numNodes(); ++j)
    {
      h = std::min(h, distance(elem.node(i), elem.node(j)));
    }
  }
  return std::isfinite(h) ? h : 0.0;
}

std::string stepLogLine(Index step,
                        Real  time,
                        Real  max_cfl,
                        bool  show_velocity_change,
                        Real  vel_change,
                        Real  assembly_sec,
                        Real  solve_sec)
{
  std::ostringstream line;
  line << "step " << std::setw(7) << step << ", t = " << std::setw(11) << time
       << ", max CFL = " << std::setw(11) << max_cfl;
  if (show_velocity_change)
  {
    line << ", rel du = " << std::setw(11) << vel_change;
  }
  line << ", assembly = " << std::setw(11) << assembly_sec << " s"
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
void packVelocity(const Vector<Real>& ux,
                  const Vector<Real>& uy,
                  const Vector<Real>& uz,
                  Vector<Real>&       velocity)
{
  const Index num_nodes = ux.size();
  if (uy.size() != num_nodes
      || uz.size() != num_nodes
      || velocity.size() != 3 * num_nodes)
  {
    throw std::runtime_error("Velocity field output received inconsistent sizes");
  }

  for (Index in = 0; in < num_nodes; ++in)
  {
    velocity[3 * in]     = ux[in];
    velocity[3 * in + 1] = uy[in];
    velocity[3 * in + 2] = uz[in];
  }
}

std::string stepVtuFile(const std::string& directory,
                        Index              level)
{
  std::ostringstream fname;
  fname << directory << "/fields_"
        << std::setw(6) << std::setfill('0') << level
        << ".vtu";
  return fname.str();
}
#endif

} // namespace

struct ForwardSolveMonitor::FieldOutput
{
  explicit FieldOutput(const fem::Mesh& mesh)
    : velocity(3 * mesh.numNodes()),
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
  Vector<Real>      velocity;
  Vector<Real>      ux;
  Vector<Real>      uy;
  Vector<Real>      uz;
  Vector<Real>      p;
};

ForwardSolveMonitor::ForwardSolveMonitor(const fem::MixedFESpace& space,
                                         Real                     dt,
                                         Index                    steps)
  : space_(&space),
    dt_(dt),
    num_steps_(steps)
{
}

ForwardSolveMonitor::~ForwardSolveMonitor() = default;

void ForwardSolveMonitor::setFieldOutput(std::string dir,
                                         Index       interval)
{
  field_dir_      = std::move(dir);
  field_interval_ = interval;
}

void ForwardSolveMonitor::clearFieldOutput()
{
  field_out_.reset();
  field_dir_.clear();
  field_interval_  = 0;
  last_field_step_ = 0;
}

void ForwardSolveMonitor::setDiagnosticsCsv(std::string file_name)
{
  diag_file_name_ = std::move(file_name);
}

void ForwardSolveMonitor::clearDiagnosticsCsv()
{
  diag_file_name_.clear();
  diag_file_.reset();
  diag_out_ = nullptr;
}

void ForwardSolveMonitor::setProgress(std::ostream* out,
                                      std::string   label)
{
  prog_out_   = out;
  prog_label_ = std::move(label);
}

void ForwardSolveMonitor::setProgressLabel(std::string label)
{
  prog_label_ = std::move(label);
}

void ForwardSolveMonitor::clearProgress()
{
  prog_out_ = nullptr;
}

void ForwardSolveMonitor::setDetailedLog(std::ostream* terminal,
                                         std::ostream* log_out,
                                         bool          show_velocity_change)
{
  log_terminal_    = terminal;
  log_out_         = log_out;
  show_vel_change_ = show_velocity_change;
}

void ForwardSolveMonitor::clearDetailedLog()
{
  log_terminal_ = nullptr;
  log_out_      = nullptr;
}

void ForwardSolveMonitor::setConvergence(ForwardConvergenceParams params)
{
  conv_ = params;
}

void ForwardSolveMonitor::clearConvergence()
{
  conv_ = ForwardConvergenceParams{};
}

const ForwardSolveResult& ForwardSolveMonitor::result() const
{
  return result_;
}

void ForwardSolveMonitor::start(Index num_steps,
                                Index num_states)
{
  (void) num_states;

  if (num_steps_ <= 0)
  {
    num_steps_ = num_steps;
  }
  result_            = ForwardSolveResult{};
  result_.vel_change = std::numeric_limits<Real>::quiet_NaN();
  last_field_step_   = 0;

  if (fieldOutputEnabled())
  {
    runtime::ensureDirectory(field_dir_);
    field_out_ = std::make_unique<FieldOutput>(space_->mesh());
  }
  if (diagnosticsEnabled())
  {
    runtime::ensureParentDirectory(diag_file_name_);
    auto file = std::make_unique<std::ofstream>(diag_file_name_);
    if (!*file)
    {
      throw std::runtime_error("Failed to open diagnostics file: "
                               + diag_file_name_);
    }
    diag_out_  = file.get();
    diag_file_ = std::move(file);
    writeDiagnosticsHeader();
  }
}

void ForwardSolveMonitor::observe(Index               level,
                                  const Vector<Real>& state)
{
  if (level > 0)
  {
    result_.final_step  = level;
    result_.final_time  = static_cast<Real>(level) * dt_;
    result_.final_state = state;
    if (fieldOutputEnabled()
        && shouldWriteForwardOutput(level, num_steps_, field_interval_))
    {
      writeFieldOutput(level, state, result_.final_time);
    }
  }
  writeDiagnosticsRow(level,
                      state,
                      static_cast<Real>(level) * dt_);
}

bool ForwardSolveMonitor::observeStep(const state::TimeStepStateContext& ctx)
{
  result_.final_step  = ctx.level;
  result_.final_time  = static_cast<Real>(ctx.level) * dt_;
  result_.final_state = ctx.current;

  const bool need_velocity_change = conv_.enabled || show_vel_change_;
  if (need_velocity_change)
  {
    result_.vel_change = velocityRelativeChange(*space_, ctx.previous, ctx.current);
  }

  result_.converged = conv_.enabled
                      && ctx.level >= conv_.min_steps
                      && result_.vel_change < conv_.vel_rel_tol;

  if (fieldOutputEnabled() && shouldWriteForwardOutput(ctx.level, num_steps_, field_interval_))
  {
    writeFieldOutput(ctx.level, ctx.current, result_.final_time);
  }

  writeDiagnosticsRow(ctx.level, ctx.current, result_.final_time);
  writeProgress(ctx.level, ctx.total_steps);

  if (detailedLogEnabled()
      && shouldWriteDetailedLog(ctx.level, ctx.total_steps))
  {
    const Real max_cfl = maxVelocityCfl(*space_, ctx.previous, dt_);
    if (!std::isfinite(max_cfl))
    {
      throw std::runtime_error("Stopping as CFL became invalid");
    }
    writeDetailedStepLog(ctx.level,
                         result_.final_time,
                         max_cfl,
                         result_.vel_change,
                         ctx.assembly_sec,
                         ctx.solve_sec);
  }

  return result_.converged;
}

void ForwardSolveMonitor::stop()
{
  writeFinalFieldOutput();
  if (diag_out_ != nullptr)
  {
    diag_out_->flush();
  }
  diag_file_.reset();
  diag_out_ = nullptr;
}

bool ForwardSolveMonitor::fieldOutputEnabled() const
{
  return !field_dir_.empty() && field_interval_ > 0;
}

bool ForwardSolveMonitor::diagnosticsEnabled() const
{
  return !diag_file_name_.empty();
}

bool ForwardSolveMonitor::detailedLogEnabled() const
{
  return log_terminal_ != nullptr || log_out_ != nullptr;
}

bool ForwardSolveMonitor::shouldWriteDetailedLog(Index step,
                                                 Index total) const
{
  if (fieldOutputEnabled())
  {
    return shouldWriteForwardOutput(step, total, field_interval_);
  }
  return true;
}

void ForwardSolveMonitor::writeFieldOutput(Index               level,
                                           const Vector<Real>& state,
                                           Real                time)
{
  if (field_out_ == nullptr)
  {
    return;
  }

  splitStateFields(VectorView<const Real>(state.data(), state.size()),
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
               field_out_->velocity);
  field_out_->vtu_out.writePointData(
      stepVtuFile(field_dir_, level),
      space_->mesh(),
      Vector<VtuWriter::PointField>{
          {"velocity", 3, &field_out_->velocity},
          {"pressure", 1, &field_out_->p}});
#endif
  last_field_step_ = level;
}

void ForwardSolveMonitor::writeFinalFieldOutput()
{
  if (!fieldOutputEnabled()
      || result_.final_step <= 0
      || last_field_step_ == result_.final_step)
  {
    return;
  }
  writeFieldOutput(result_.final_step,
                   result_.final_state,
                   result_.final_time);
}

void ForwardSolveMonitor::writeDiagnosticsHeader()
{
  if (diag_out_ != nullptr)
  {
    *diag_out_ << "level,time,state_l2\n";
  }
}

void ForwardSolveMonitor::writeDiagnosticsRow(Index               level,
                                              const Vector<Real>& state,
                                              Real                time)
{
  if (diag_out_ == nullptr)
  {
    return;
  }

  const auto prec = diag_out_->precision();
  *diag_out_ << std::setprecision(std::numeric_limits<Real>::digits10 + 1)
             << level << ',' << time << ',' << norm(state) << '\n';
  diag_out_->precision(prec);
}

void ForwardSolveMonitor::writeProgress(Index step,
                                        Index total)
{
  if (prog_out_ == nullptr)
  {
    return;
  }

  const Index stride = std::max<Index>(Index{1}, total / 5);
  if (step % stride != 0 && step < total)
  {
    return;
  }

  *prog_out_ << "\r    " << prog_label_ << " step "
             << std::setw(4) << step << " / "
             << std::setw(4) << total << std::flush;
  if (step >= total)
  {
    *prog_out_ << '\n';
  }
}

void ForwardSolveMonitor::writeDetailedStepLog(Index step,
                                               Real  time,
                                               Real  max_cfl,
                                               Real  vel_change,
                                               Real  assembly_sec,
                                               Real  solve_sec)
{
  writeLine(stepLogLine(step,
                        time,
                        max_cfl,
                        show_vel_change_,
                        vel_change,
                        assembly_sec,
                        solve_sec),
            log_terminal_,
            log_out_);
}

Real velocityRelativeChange(const fem::MixedFESpace& space,
                            const Vector<Real>&      previous,
                            const Vector<Real>&      current)
{
  if (previous.size() != current.size()
      || previous.size() != space.numDofs())
  {
    throw std::runtime_error("velocity convergence received incompatible states");
  }

  const auto  velocity  = space.field(0);
  const Index num_nodes = velocity.space().mesh().numNodes();
  const Index comps     = velocity.numComponents();

  Real diff2 = 0.0;
  Real ref2  = 0.0;
  for (Index in = 0; in < num_nodes; ++in)
  {
    for (Index d = 0; d < comps; ++d)
    {
      const Index id    = velocity.globalDof(in, d);
      const Real  diff  = current[id] - previous[id];
      diff2            += diff * diff;
      ref2             += previous[id] * previous[id];
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
                    const Vector<Real>&      state,
                    Real                     dt)
{
  if (state.size() != space.numDofs())
  {
    throw std::runtime_error("CFL calculation received incompatible state size");
  }

  const auto  velocity = space.field(0);
  const Index comps    = velocity.numComponents();
  Real        max_cfl  = 0.0;

  for (Index ie = 0; ie < space.mesh().numElems(); ++ie)
  {
    const fem::Element& elem = space.mesh().elem(ie);
    const Real          h    = elemMinEdge(elem);
    if (h <= 0.0)
    {
      continue;
    }

    for (Index in = 0; in < elem.numNodes(); ++in)
    {
      const Index id   = elem.nodeIds()[in];
      Real        vel2 = 0.0;
      for (Index d = 0; d < comps; ++d)
      {
        const Real value  = state[velocity.globalDof(id, d)];
        vel2             += value * value;
      }
      max_cfl = std::max(max_cfl, std::sqrt(vel2) * dt / h);
    }
  }

  return max_cfl;
}

bool shouldWriteForwardOutput(Index step,
                              Index total_steps,
                              Index interval)
{
  return interval > 0 && (step % interval == 0 || step == total_steps);
}

} // namespace femx::model::ns
