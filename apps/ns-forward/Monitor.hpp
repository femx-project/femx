#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "Config.hpp"
#include <femx/common/Types.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeIntegrator.hpp>

namespace femx::apps::ns_forward
{

struct SolveResult
{
  HostVector<Real> final_state;      ///< Last accepted state vector.
  Index            final_step{0};    ///< Last completed time step.
  Real             final_time{0.0};  ///< Physical time at final_step.
  Real             vel_change{0.0};  ///< Last relative velocity change.
  bool             converged{false}; ///< True when convergence stopped the run.
};

class Monitor final
{
public:
  Monitor(const fem::MixedFESpace& space,
          Real                     dt,
          Index                    steps);
  ~Monitor();

  void setFieldOutput(std::string directory,
                      Index       interval);

  void setDetailedLog(std::ostream* terminal,
                      std::ostream* log_out,
                      bool          show_velocity_change);

  void setConvergence(ConvergenceConfig params);

  const SolveResult& result() const;

  void start(Index num_steps,
             Index num_states);
  void observe(Index                   level,
               const HostVector<Real>& state);
  bool observeStep(const state::TimeStepStateContext& ctx);
  void stop();

private:
  struct FieldOutput;

  bool fieldOutputEnabled() const;
  bool detailedLogEnabled() const;
  bool shouldWriteDetailedLog(Index step,
                              Index total) const;

  void writeFieldOutput(Index                   level,
                        const HostVector<Real>& state,
                        Real                    time);
  void writeFinalFieldOutput();
  void writeDetailedStepLog(Index step,
                            Real  time,
                            Real  max_cfl,
                            Real  vel_change,
                            Real  assembly_sec,
                            Real  solve_sec);

private:
  const fem::MixedFESpace*     space_{nullptr};
  Real                         dt_{0.0};
  Index                        num_steps_{0};
  SolveResult                  result_;
  ConvergenceConfig            convergence_;
  std::unique_ptr<FieldOutput> field_out_;
  std::string                  field_dir_;
  Index                        field_interval_{0};
  Index                        last_field_step_{0};
  std::ostream*                log_terminal_{nullptr};
  std::ostream*                log_out_{nullptr};
  bool                         show_vel_change_{false};
};

Real velocityRelativeChange(const fem::MixedFESpace& space,
                            const HostVector<Real>&  prev,
                            const HostVector<Real>&  curr);

Real maxVelocityCfl(const fem::MixedFESpace& space,
                    const HostVector<Real>&  state,
                    Real                     dt);

bool shouldWriteOutput(Index step,
                       Index total_steps,
                       Index interval);

} // namespace femx::apps::ns_forward
