#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/TimeProblemAdapter.hpp>
#include <femx/problem/TimeResidualEquation.hpp>
#include <femx/solve/TimeStateSolver.hpp>
#include <femx/solve/TimeStateTrajectory.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/solve/TimeStepper.hpp>
#include <femx/algebra/LinearSolver.hpp>

namespace femx
{
namespace solve
{

using problem::TimeResidualEquation;
using problem::TimeResidualEquationProblemAdapter;

struct TimeNewtonStateSolverOptions
{
  Index max_its        = 20;
  Real  res_tol        = 1.0e-10;
  Real  step_tolerance = 0.0;
};

/**
 * @brief Compatibility solver. Prefer solve::TimeStepper for new code.
 *
 * The initial state defaults to zero. Each next-state Newton iteration starts
 * from the previous time level, which is a practical default for parabolic and
 * flow time stepping.
 */
class TimeNewtonStateSolver final : public TimeStateSolver
{
public:
  TimeNewtonStateSolver(const TimeResidualEquation& eq,
                        algebra::LinearSolver&       lin_solver);

  TimeNewtonStateSolverOptions& options();

  const TimeNewtonStateSolverOptions& options() const;

  void setInitialState(const Vector<Real>& state);

  void clearInitialState();

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  void solve(const Vector<Real>&  prm,
             TimeStateTrajectory& tr) override;

private:
  void syncOptions();

private:
  TimeResidualEquationProblemAdapter problem_;
  solve::TimeStepper                 stepper_;
  TimeNewtonStateSolverOptions       options_;
};

} // namespace solve
} // namespace femx
