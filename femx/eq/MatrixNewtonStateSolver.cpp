#include <stdexcept>

#include <femx/common/Math.hpp>
#include <femx/eq/MatrixNewtonStateSolver.hpp>

using namespace femx::system;

namespace femx
{
namespace eq
{

MatrixNewtonStateSolver::MatrixNewtonStateSolver(
    const MatrixResidualEquation& eq,
    SystemMatrix&         state_jac,
    LinearSolver&         lin_solver)
  : eq_(eq),
    state_jac_(state_jac),
    lin_solver_(lin_solver)
{
  if (eq_.numRes() != eq_.numStates())
  {
    throw std::runtime_error(
        "MatrixNewtonStateSolver requires square state residual dimensions");
  }
}

MatrixNewtonStateSolverOptions& MatrixNewtonStateSolver::options()
{
  return options_;
}

const MatrixNewtonStateSolverOptions& MatrixNewtonStateSolver::options() const
{
  return options_;
}

void MatrixNewtonStateSolver::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error(
        "MatrixNewtonStateSolver initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void MatrixNewtonStateSolver::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

Index MatrixNewtonStateSolver::numStates() const
{
  return eq_.numStates();
}

Index MatrixNewtonStateSolver::numParams() const
{
  return eq_.numParams();
}

void MatrixNewtonStateSolver::solve(const Vector<Real>& prm,
                                    Vector<Real>&       state)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "MatrixNewtonStateSolver parameter size mismatch");
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
      throw std::runtime_error("MatrixNewtonStateSolver residual size mismatch");
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

    eq_.assembleStateJac(state, prm, state_jac_);
    state_jac_.finalize();
    lin_solver_.solve(state_jac_, rhs, step);
    if (step.size() != numStates())
    {
      throw std::runtime_error("MatrixNewtonStateSolver step size mismatch");
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

  throw std::runtime_error("MatrixNewtonStateSolver failed to converge");
}

void MatrixNewtonStateSolver::initializeState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resize(state, numStates());
}

void MatrixNewtonStateSolver::resize(Vector<Real>& out,
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
