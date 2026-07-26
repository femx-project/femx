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

namespace femx::apps::navier
{

/** @brief Store the final state and convergence summary of a forward run. */
struct SolveResult
{
  HostVector<Real> final_state;      ///< Last accepted state vector.
  Index            final_step{0};    ///< Last completed time step.
  Real             final_time{0.0};  ///< Physical time at final_step.
  Real             vel_change{0.0};  ///< Last relative velocity change.
  bool             converged{false}; ///< True when convergence stopped the run.
};

/** @brief Monitor time-step progress, convergence, logging, and field output. */
class Monitor final
{
public:
  /**
   * @brief Construct a monitor for one finite-element time interval.
   *
   * @param[in] space - State finite-element space.
   * @param[in] dt - Time-step size.
   * @param[in] steps - Configured number of time steps.
   */
  Monitor(const fem::MixedFESpace& space,
          Real                     dt,
          Index                    steps);
  ~Monitor();

  /**
   * @brief Configure field output.
   *
   * @param[in] dir - Output directory.
   * @param[in] interval - Time-step interval between output levels.
   */
  void setFieldOutput(std::string dir,
                      Index       interval);

  /**
   * @brief Configure terminal and persistent detailed logs.
   *
   * @param[out] terminal - Optional terminal stream.
   * @param[out] log_out - Optional persistent log stream.
   * @param[in] show_velocity_change - Include the relative velocity change.
   */
  void setDetailedLog(std::ostream* terminal,
                      std::ostream* log_out,
                      bool          show_velocity_change);

  /**
   * @brief Configure the steady-state convergence criterion.
   *
   * @param[in] prm - Convergence configuration.
   */
  void setConvergence(ConvergenceConfig prm);

  /** @brief Return the current solve result. */
  const SolveResult& result() const;

  /**
   * @brief Start monitoring a time integration.
   *
   * @param[in] num_steps - Integrator step count.
   * @param[in] num_states - Integrator state size.
   */
  void start(Index num_steps,
             Index num_states);

  /**
   * @brief Observe one accepted state level.
   *
   * @param[in] level - Accepted time level.
   * @param[in] state - Accepted state.
   */
  void observe(Index                   level,
               const HostVector<Real>& state);

  /**
   * @brief Observe a completed time step.
   *
   * @param[in] ctx - Completed-step context.
   * @return `true` when the convergence criterion is satisfied.
   */
  bool observeStep(const state::TimeStepStateContext& ctx);

  /** @brief Stop monitoring and write the final field level if needed. */
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
                            Real  assm_sec,
                            Real  solve_sec);

private:
  const fem::MixedFESpace*     space_{nullptr};         ///< State finite-element space.
  Real                         dt_{0.0};                ///< Time-step size.
  Index                        num_steps_{0};           ///< Number of time steps.
  SolveResult                  res_;                    ///< Current solve result.
  ConvergenceConfig            convergence_;            ///< Convergence settings.
  std::unique_ptr<FieldOutput> field_out_;              ///< Field-output state.
  std::string                  field_dir_;              ///< Field-output directory.
  Index                        field_interval_{0};      ///< Field-output interval.
  Index                        last_field_step_{0};     ///< Last written field step.
  std::ostream*                log_terminal_{nullptr};  ///< Terminal log stream.
  std::ostream*                log_out_{nullptr};       ///< Persistent log stream.
  bool                         show_vel_change_{false}; ///< Log velocity changes.
};

/**
 * @brief Compute the relative velocity change between two states.
 *
 * @param[in] space - State finite-element space.
 * @param[in] prev - Previous state.
 * @param[in] curr - Current state.
 * @return Relative velocity change.
 * @throws std::runtime_error - If the state sizes are incompatible.
 */
Real velocityRelativeChange(const fem::MixedFESpace& space,
                            const HostVector<Real>&  prev,
                            const HostVector<Real>&  curr);

/**
 * @brief Compute the maximum element velocity CFL number.
 *
 * @param[in] space - State finite-element space.
 * @param[in] state - State used for the velocity field.
 * @param[in] dt - Time-step size.
 * @return Maximum velocity CFL number.
 * @throws std::runtime_error - If the state size is incompatible.
 */
Real maxVelocityCfl(const fem::MixedFESpace& space,
                    const HostVector<Real>&  state,
                    Real                     dt);

/**
 * @brief Return whether a step should be written.
 *
 * @param[in] step - Current time step.
 * @param[in] total_steps - Final time step.
 * @param[in] interval - Output interval.
 * @return `true` at an output interval or the final step.
 */
bool shouldWriteOutput(Index step,
                       Index total_steps,
                       Index interval);

} // namespace femx::apps::navier
