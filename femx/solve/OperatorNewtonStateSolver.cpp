#include <stdexcept>

#include <femx/solve/OperatorNewtonStateSolver.hpp>
#include <femx/problem/ProblemResidualAdapter.hpp>
#include <femx/solve/Newton.hpp>

using namespace femx::algebra;

namespace femx
{
namespace solve
{

OperatorNewtonStateSolver::OperatorNewtonStateSolver(
    const ResidualEquation& eq,
    LinearSolver&           lin_solver)
  : eq_(eq),
    lin_solver_(lin_solver)
{
  if (eq_.numRes() != eq_.numStates())
  {
    throw std::runtime_error(
        "OperatorNewtonStateSolver requires square state residual dimensions");
  }
}

OperatorNewtonStateSolverOptions& OperatorNewtonStateSolver::options()
{
  return options_;
}

const OperatorNewtonStateSolverOptions& OperatorNewtonStateSolver::options()
    const
{
  return options_;
}

void OperatorNewtonStateSolver::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error(
        "OperatorNewtonStateSolver initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void OperatorNewtonStateSolver::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

Index OperatorNewtonStateSolver::numStates() const
{
  return eq_.numStates();
}

Index OperatorNewtonStateSolver::numParams() const
{
  return eq_.numParams();
}

void OperatorNewtonStateSolver::solve(const Vector<Real>& prm,
                                      Vector<Real>&       state)
{
  problem::ResidualEquationProblemAdapter adapter(eq_);
  problem::ResidualEquationLinearization  linearization;
  solve::Newton                   newton(adapter, linearization, lin_solver_);
  newton.options().max_its            = options_.max_its;
  newton.options().residual_tolerance = options_.res_tol;
  newton.options().step_tolerance     = options_.step_tolerance;
  if (has_init_state_)
  {
    newton.setInitialState(init_state_);
  }
  newton.solve(prm, state);
}

void OperatorNewtonStateSolver::initializeState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resize(state, numStates());
}

void OperatorNewtonStateSolver::resize(Vector<Real>& out,
                                       Index         size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

} // namespace solve
} // namespace femx
