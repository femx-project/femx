#pragma once

#include <chrono>
#include <iosfwd>
#include <memory>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeStateMonitor.hpp>

namespace femx::model::ns
{

using Clock = std::chrono::high_resolution_clock;

struct ForwardSolveResult
{
  Vector<Real> final_state;
  Index        final_step{0};
  Real         final_time{0.0};
  Real         vel_change{0.0};
  bool         converged{false};
};

struct ForwardConvergenceParams
{
  bool  enabled     = false;
  Real  vel_rel_tol = 1.0e-8;
  Index min_steps   = 1;
};

class ForwardSolveMonitor final : public state::TimeStateMonitor
{
public:
  ForwardSolveMonitor(const MixedFESpace& space,
                      Real                dt,
                      Index               steps);
  ~ForwardSolveMonitor() override;

  void setFieldOutput(std::string directory,
                      Index       interval);
  void clearFieldOutput();

  void setDiagnosticsCsv(std::string file_name);
  void clearDiagnosticsCsv();

  void setProgress(std::ostream* out, std::string label = "time");
  void setProgressLabel(std::string label);
  void clearProgress();

  void setDetailedLog(std::ostream* terminal,
                      std::ostream* log_out,
                      bool          show_velocity_change);
  void clearDetailedLog();

  void setConvergence(ForwardConvergenceParams params);
  void clearConvergence();

  const ForwardSolveResult& result() const;

  void start(Index num_steps,
             Index num_states) override;
  void observe(Index               level,
               const Vector<Real>& state) override;
  bool observeStep(const state::TimeStepStateContext& ctx) override;
  void stop() override;

private:
  struct FieldOutput;

  bool fieldOutputEnabled() const;
  bool diagnosticsEnabled() const;
  bool detailedLogEnabled() const;
  bool shouldWriteDetailedLog(Index step,
                              Index total) const;

  void writeFieldOutput(Index               level,
                        const Vector<Real>& state,
                        Real                time);
  void writeFinalFieldOutput();
  void writeDiagnosticsHeader();
  void writeDiagnosticsRow(Index               level,
                           const Vector<Real>& state,
                           Real                time);
  void writeProgress(Index step,
                     Index total);
  void writeDetailedStepLog(Index step,
                            Real  time,
                            Real  max_cfl,
                            Real  vel_change,
                            Real  assembly_seconds,
                            Real  solve_seconds);

private:
  const MixedFESpace*            space_{nullptr};
  Real                           dt_{0.0};
  Index                          num_steps_{0};
  ForwardSolveResult             result_;
  ForwardConvergenceParams       conv_;
  std::unique_ptr<FieldOutput>   field_out_;
  std::string                    field_dir_;
  Index                          field_interval_{0};
  Index                          last_field_step_{0};
  std::string                    diag_file_name_;
  std::unique_ptr<std::ofstream> diag_file_;
  std::ostream*                  diag_out_{nullptr};
  std::ostream*                  prog_out_{nullptr};
  std::string                    prog_label_{"time"};
  std::ostream*                  log_terminal_{nullptr};
  std::ostream*                  log_out_{nullptr};
  bool                           show_vel_change_{false};
  Clock::time_point              step_begin_;
};

Real velocityRelativeChange(const MixedFESpace& space,
                            const Vector<Real>& previous,
                            const Vector<Real>& current);

Real maxVelocityCfl(const MixedFESpace& space,
                    const Vector<Real>& state,
                    Real                dt);

bool shouldWriteForwardOutput(Index step,
                              Index total_steps,
                              Index interval);

} // namespace femx::model::ns
