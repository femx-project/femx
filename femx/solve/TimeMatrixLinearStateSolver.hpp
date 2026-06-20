#pragma once

#include <functional>

#include <femx/core/Types.hpp>
#include <femx/problem/TimeMatrixResidualEquation.hpp>
#include <femx/solve/TimeStateSolver.hpp>
#include <femx/solve/TimeStateTrajectory.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearSolver.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace solve
{

using problem::TimeMatrixResidualEquation;

/** @brief Time-marching solver for residual equations affine in the next state.
 *
 * At each time step this solves one matrix linearization
 *
 *   R_{u_next}(u_next_ref, u_previous, m) du
 *       = -R(u_next_ref, u_previous, m),
 *   u_next = u_next_ref + du.
 *
 * For equations affine in u_next this is the exact state solve.
 */
class TimeMatrixLinearStateSolver final : public TimeStateSolver
{
public:
  using StepMonitor = std::function<void(Index step, Index total_steps)>;

  TimeMatrixLinearStateSolver(const TimeMatrixResidualEquation& eq,
                              algebra::SystemMatrix&             next_state_jac,
                              algebra::LinearSolver&             lin_solver);

  void setInitialState(const Vector<Real>& state);

  void clearInitialState();

  void setStepMonitor(StepMonitor monitor);

  void clearStepMonitor();

  void resetTiming();

  Real assemblySeconds() const;

  Real solveSeconds() const;

  Index assemblyCalls() const;

  Index solveCalls() const;

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  void solve(const Vector<Real>&  prm,
             TimeStateTrajectory& tr) override;

private:
  void solveStep(Index               step,
                 const Vector<Real>& prm,
                 const Vector<Real>& x,
                 Vector<Real>&       x_next);

  void initializeInitialState(Vector<Real>& state) const;

  static void resize(Vector<Real>& out,
                     Index         size);

private:
  const TimeMatrixResidualEquation& eq_;
  algebra::SystemMatrix&             next_state_jac_;
  algebra::LinearSolver&             lin_solver_;
  Vector<Real>                      init_state_;
  StepMonitor                       step_monitor_;
  Real                              assembly_seconds_{0.0};
  Real                              solve_seconds_{0.0};
  Index                             assembly_calls_{0};
  Index                             solve_calls_{0};
  bool                              has_init_state_{false};
};

} // namespace solve
} // namespace femx
