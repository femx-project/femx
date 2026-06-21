#pragma once

#include <functional>

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/solve/TimeStateSolver.hpp>
#include <femx/solve/TimeTrajectory.hpp>

namespace femx
{
namespace solve
{

/** @brief Time-marching solver using the assembled next-state Jacobian. */
class TimeLinearStateSolver final : public TimeStateSolver
{
public:
  using StepMonitor = std::function<void(Index step, Index total_steps)>;

  TimeLinearStateSolver(const problem::TimeResidual& eq,
                        linalg::MatrixOperator&      next_state_jac,
                        linalg::LinearSolver&        lin_solver);

  void setInitialState(const Vector<Real>& state);
  void clearInitialState();

  void setStepMonitor(StepMonitor monitor);
  void clearStepMonitor();

  void  resetTiming();
  Real  assemblySeconds() const;
  Real  solveSeconds() const;
  Index assemblyCalls() const;
  Index solveCalls() const;

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  void solve(const Vector<Real>& prm,
             TimeTrajectory&     tr) override;

private:
  void solveStep(Index               step,
                 const Vector<Real>& prm,
                 const Vector<Real>& x,
                 Vector<Real>&       x_next);

  void        initializeInitialState(Vector<Real>& state) const;
  static void resize(Vector<Real>& out, Index size);

private:
  const problem::TimeResidual& eq_;
  linalg::MatrixOperator&      next_state_jac_;
  linalg::LinearSolver&        lin_solver_;
  problem::TimeDims      dims_;
  Vector<Real>                 init_state_;
  StepMonitor                  step_monitor_;
  Real                         assembly_seconds_{0.0};
  Real                         solve_seconds_{0.0};
  Index                        assembly_calls_{0};
  Index                        solve_calls_{0};
  bool                         has_init_state_{false};
};

} // namespace solve
} // namespace femx
