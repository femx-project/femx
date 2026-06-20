#include <stdexcept>

#include <femx/common/Math.hpp>
#include <femx/eq/OperatorNewtonStateSolver.hpp>
#include <femx/eq/StateJacobianOperator.hpp>

using namespace femx::system;

namespace femx
{
namespace eq
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
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "OperatorNewtonStateSolver parameter size mismatch");
  }

  initializeState(state);

  Vector<Real> res;
  Vector<Real> rhs;
  Vector<Real> step;
  for (Index i = 0; i <= options_.max_its; ++i)
  {
    eq_.res(state, prm, res);
    if (res.size() != eq_.numRes())
    {
      throw std::runtime_error(
          "OperatorNewtonStateSolver residual size mismatch");
    }

    if (squaredNorm(res) <= options_.res_tol * options_.res_tol)
    {
      return;
    }
    if (i == options_.max_its)
    {
      break;
    }

    resize(rhs, res.size());
    for (Index i = 0; i < res.size(); ++i)
    {
      rhs[i] = -res[i];
    }

    const StateJacobianOperator jac(eq_, state, prm);
    lin_solver_.solve(jac, rhs, step);
    if (step.size() != numStates())
    {
      throw std::runtime_error("OperatorNewtonStateSolver step size mismatch");
    }

    for (Index i = 0; i < numStates(); ++i)
    {
      state[i] += step[i];
    }

    if (squaredNorm(step) <= options_.step_tolerance * options_.step_tolerance)
    {
      return;
    }
  }

  throw std::runtime_error("OperatorNewtonStateSolver failed to converge");
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

} // namespace eq
} // namespace femx
