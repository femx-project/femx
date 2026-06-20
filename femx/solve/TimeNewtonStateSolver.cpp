#include <femx/solve/TimeNewtonStateSolver.hpp>

namespace femx
{
namespace solve
{

TimeNewtonStateSolver::TimeNewtonStateSolver(
    const TimeResidualEquation& eq,
    algebra::LinearSolver&       lin_solver)
  : problem_(eq),
    stepper_(problem_, lin_solver)
{
}

TimeNewtonStateSolverOptions& TimeNewtonStateSolver::options()
{
  return options_;
}

const TimeNewtonStateSolverOptions& TimeNewtonStateSolver::options() const
{
  return options_;
}

void TimeNewtonStateSolver::setInitialState(const Vector<Real>& state)
{
  stepper_.setInitialState(state);
}

void TimeNewtonStateSolver::clearInitialState()
{
  stepper_.clearInitialState();
}

Index TimeNewtonStateSolver::numSteps() const
{
  return stepper_.numSteps();
}

Index TimeNewtonStateSolver::numStates() const
{
  return stepper_.numStates();
}

Index TimeNewtonStateSolver::numParams() const
{
  return stepper_.numParams();
}

void TimeNewtonStateSolver::solve(const Vector<Real>&  prm,
                                  TimeStateTrajectory& tr)
{
  syncOptions();

  solve::TimeTrajectory solve_tr;
  stepper_.solve(prm, solve_tr);

  tr.resize(solve_tr.numSteps(), solve_tr.numStates());
  for (Index level = 0; level < solve_tr.numTimeLevels(); ++level)
  {
    tr[level] = solve_tr[level];
  }
}

void TimeNewtonStateSolver::syncOptions()
{
  stepper_.options().max_its            = options_.max_its;
  stepper_.options().residual_tolerance = options_.res_tol;
  stepper_.options().step_tolerance     = options_.step_tolerance;
}

} // namespace solve
} // namespace femx
