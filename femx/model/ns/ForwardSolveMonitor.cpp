#include "ForwardSolveMonitor.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <femx/common/Math.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/runtime/Output.hpp>

using namespace std;

namespace femx::model::ns
{
namespace
{

double elapsedSeconds(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double>(end - begin).count();
}

Real elemMinEdge(const Element& elem)
{
  Real h = numeric_limits<Real>::infinity();
  for (Index i = 0; i < elem.numNodes(); ++i)
  {
    for (Index j = i + 1; j < elem.numNodes(); ++j)
    {
      h = min(h, distance(elem.node(i), elem.node(j)));
    }
  }
  return isfinite(h) ? h : 0.0;
}

void splitFields(const Vector<Real>& x,
                 const MixedFESpace& space,
                 Vector<Real>&       ux,
                 Vector<Real>&       uy,
                 Vector<Real>&       uz,
                 Vector<Real>&       p)
{
  const Mesh& mesh  = space.mesh();
  const auto  u_dof = space.field(0);
  const auto  p_dof = space.field(1);
  const Index num_components    = u_dof.numComponents();

  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    ux[in] = x[u_dof.globalDof(in, 0)];
    uy[in] = 0.0;
    uz[in] = 0.0;
    if (num_components > 1)
    {
      uy[in] = x[u_dof.globalDof(in, 1)];
    }
    if (num_components > 2)
    {
      uz[in] = x[u_dof.globalDof(in, 2)];
    }
    p[in] = x[p_dof.globalDof(in)];
  }
}

string stepLogLine(Index step,
                   Real  time,
                   Real  max_cfl,
                   bool  show_velocity_change,
                   Real  vel_change,
                   Real  assembly_seconds,
                   Real  solve_seconds,
                   Real  total_seconds)
{
  ostringstream line;
  line << "step " << setw(7) << step << ", t = " << setw(11) << time
       << ", max CFL = " << setw(11) << max_cfl;
  if (show_velocity_change)
  {
    line << ", rel du = " << setw(11) << vel_change;
  }
  line << ", assembly = " << setw(11) << assembly_seconds << " s"
       << ", solve = " << setw(11) << solve_seconds << " s"
       << ", total = " << setw(11) << total_seconds << " s";
  return line.str();
}

void writeLine(const string& line,
               ostream*      terminal,
               ostream*      log_out)
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

} // namespace

struct ForwardSolveMonitor::FieldOutput
{
  explicit FieldOutput(const Mesh& mesh)
    : ux(mesh.numNodes()),
      uy(mesh.numNodes()),
      uz(mesh.numNodes()),
      p(mesh.numNodes())
  {
    vel_out.attachMesh(mesh);
    pre_out.attachMesh(mesh);
  }

  TimeSeriesDataOut vel_out;
  TimeSeriesDataOut pre_out;
  Vector<Real>      ux;
  Vector<Real>      uy;
  Vector<Real>      uz;
  Vector<Real>      p;
};

ForwardSolveMonitor::ForwardSolveMonitor(const MixedFESpace& space,
                                         Real                dt,
                                         Index               steps)
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
  result_.vel_change = numeric_limits<Real>::quiet_NaN();
  last_field_step_   = 0;
  step_begin_        = Clock::now();

  if (fieldOutputEnabled())
  {
    runtime::ensureDirectory(field_dir_);
    field_out_ = make_unique<FieldOutput>(space_->mesh());
  }
  if (diagnosticsEnabled())
  {
    runtime::ensureParentDirectory(diag_file_name_);
    auto file = make_unique<ofstream>(diag_file_name_);
    if (!*file)
    {
      throw runtime_error("Failed to open diagnostics file: "
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
    if (!isfinite(max_cfl))
    {
      throw runtime_error("Stopping as CFL became invalid");
    }
    writeDetailedStepLog(ctx.level,
                         result_.final_time,
                         max_cfl,
                         result_.vel_change,
                         ctx.assembly_seconds,
                         ctx.solve_seconds);
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

  splitFields(state,
              *space_,
              field_out_->ux,
              field_out_->uy,
              field_out_->uz,
              field_out_->p);

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
  *diag_out_ << setprecision(numeric_limits<Real>::digits10 + 1)
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

  const Index stride = max<Index>(Index{1}, total / 5);
  if (step % stride != 0 && step < total)
  {
    return;
  }

  *prog_out_ << "\r    " << prog_label_ << " step "
             << setw(4) << step << " / "
             << setw(4) << total << flush;
  if (step >= total)
  {
    *prog_out_ << '\n';
  }
}

void ForwardSolveMonitor::writeDetailedStepLog(Index step,
                                               Real  time,
                                               Real  max_cfl,
                                               Real  vel_change,
                                               Real  assembly_seconds,
                                               Real  solve_seconds)
{
  const Real total_seconds = elapsedSeconds(step_begin_, Clock::now());
  writeLine(stepLogLine(step,
                        time,
                        max_cfl,
                        show_vel_change_,
                        vel_change,
                        assembly_seconds,
                        solve_seconds,
                        total_seconds),
            log_terminal_,
            log_out_);
  step_begin_ = Clock::now();
}

Real velocityRelativeChange(const MixedFESpace& space,
                            const Vector<Real>& previous,
                            const Vector<Real>& current)
{
  if (previous.size() != current.size()
      || previous.size() != space.numDofs())
  {
    throw runtime_error("velocity convergence received incompatible states");
  }

  const auto  velocity = space.field(0);
  const Index num_nodes    = velocity.space().mesh().numNodes();
  const Index comps    = velocity.numComponents();

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
    return numeric_limits<Real>::infinity();
  }
  return sqrt(diff2 / ref2);
}

Real maxVelocityCfl(const MixedFESpace& space,
                    const Vector<Real>& state,
                    Real                dt)
{
  if (state.size() != space.numDofs())
  {
    throw runtime_error("CFL calculation received incompatible state size");
  }

  const auto  velocity = space.field(0);
  const Index comps    = velocity.numComponents();
  Real        max_cfl  = 0.0;

  for (Index ie = 0; ie < space.mesh().numElems(); ++ie)
  {
    const Element& elem = space.mesh().elem(ie);
    const Real     h    = elemMinEdge(elem);
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
      max_cfl = max(max_cfl, sqrt(vel2) * dt / h);
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
