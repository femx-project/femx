#pragma once

#include <functional>

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/state/TimeStateSolver.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace state
{

/** @brief Time-marching solver using the assembled next-state Jacobian. */
class TimeLinearStateSolver final : public TimeStateSolver
{
public:
  using StepMonitor   = std::function<void(Index step, Index total_steps)>;
  using StateObserver = std::function<void(Index level, const Vector<Real>& state)>;
  using StopCondition = std::function<bool(Index               level,
                                           const Vector<Real>& current,
                                           const Vector<Real>& previous)>;

  TimeLinearStateSolver(const problem::TimeResidual& problem,
                        linalg::MatrixOperator&      J_next,
                        linalg::LinearSolver&        lin_solver);

  void setInitialState(const Vector<Real>& state);
  void clearInitialState();

  void setStepMonitor(StepMonitor monitor);
  void clearStepMonitor();

  void setStopCondition(StopCondition condition);
  void clearStopCondition();

  void  resetTiming();
  Real  assemblySeconds() const;
  Real  solveSeconds() const;
  Real  lastAssemblySeconds() const;
  Real  lastSolveSeconds() const;
  Index assemblyCalls() const;
  Index solveCalls() const;

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  void solve(const Vector<Real>& prm,
             TimeTrajectory&     tr) override;

  void solve(const Vector<Real>&  prm,
             const StateObserver& observer);

private:
  void solveStep(Index               step,
                 const Vector<Real>& prm,
                 const Vector<Real>& hist,
                 Vector<Real>&       x_next);

  void initializeInitialState(Vector<Real>& state) const;

private:
  const problem::TimeResidual& problem_;
  linalg::MatrixOperator&      J_next_;
  linalg::LinearSolver&        lin_solver_;
  problem::TimeDims            dims_;
  Vector<Real>                 init_state_;
  StepMonitor                  step_monitor_;
  StopCondition                stop_condition_;
  Real                         assembly_seconds_{0.0};
  Real                         solve_sec_{0.0};
  Real                         last_assm_sec_{0.0};
  Real                         last_solve_sec_{0.0};
  Index                        assm_calls_{0};
  Index                        solve_calls_{0};
  bool                         has_init_state_{false};
};

} // namespace state
} // namespace femx
