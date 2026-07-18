#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx
{
namespace linalg
{
class MatrixOperator;
class LinearSolver;
} // namespace linalg

namespace state
{

class TimeTrajectory;

/**
 * @brief Time integrator using the assembled next-state Jacobian.
 *
 * TimeLinearIntegrator assembles the next-state Jacobian at each step and
 * solves for the next state with a linear solver.
 */
class TimeLinearIntegrator final : public TimeIntegrator
{
public:
  TimeLinearIntegrator(const state::TimeResidual& problem,
                       linalg::MatrixOperator&    J_next,
                       linalg::LinearSolver&      lin_solver);

  void setInitialState(const HostVector& state);
  void clearInitialState();

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

  void solve(const HostVector& prm,
             TimeTrajectory&   tr) override;

  void solve(const HostVector& prm);

private:
  void solveImpl(const HostVector& prm,
                 TimeTrajectory*   tr);

  void solveStep(Index             step,
                 const HostVector& prm,
                 const HostVector& hist,
                 HostVector&       x_next);

  void initializeInitialState(HostVector& state) const;

private:
  const state::TimeResidual& problem_;
  linalg::MatrixOperator&    J_next_;
  linalg::LinearSolver&      lin_solver_;
  state::TimeDims            dims_;
  HostVector                 init_state_;
  Real                       assm_sec_{0.0};
  Real                       solve_sec_{0.0};
  Real                       last_assm_sec_{0.0};
  Real                       last_solve_sec_{0.0};
  Index                      assm_calls_{0};
  Index                      solve_calls_{0};
  bool                       has_init_state_{false};
};

} // namespace state
} // namespace femx
