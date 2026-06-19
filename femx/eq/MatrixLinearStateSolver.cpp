#include <stdexcept>

#include <femx/eq/MatrixLinearStateSolver.hpp>

using namespace femx::system;

namespace femx
{
namespace eq
{

MatrixLinearStateSolver::MatrixLinearStateSolver(
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
        "MatrixLinearStateSolver requires square state residual dimensions");
  }
}

void MatrixLinearStateSolver::setReferenceState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error(
        "MatrixLinearStateSolver reference state size mismatch");
  }
  reference_state_     = state;
  has_reference_state_ = true;
}

void MatrixLinearStateSolver::clearReferenceState()
{
  reference_state_     = Vector<Real>{};
  has_reference_state_ = false;
}

Index MatrixLinearStateSolver::numStates() const
{
  return eq_.numStates();
}

Index MatrixLinearStateSolver::numParams() const
{
  return eq_.numParams();
}

void MatrixLinearStateSolver::solve(const Vector<Real>& prm,
                                    Vector<Real>&       state)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "MatrixLinearStateSolver parameter size mismatch");
  }

  Vector<Real> reference;
  initializeReferenceState(reference);

  Vector<Real> rhs;
  eq_.res(reference, prm, rhs);
  if (rhs.size() != eq_.numRes())
  {
    throw std::runtime_error("MatrixLinearStateSolver residual size mismatch");
  }

  for (Index i = 0; i < rhs.size(); ++i)
  {
    rhs[i] = -rhs[i];
  }

  eq_.assembleStateJac(reference, prm, state_jac_);
  state_jac_.finalize();

  Vector<Real> step;
  lin_solver_.solve(state_jac_, rhs, step);
  if (step.size() != numStates())
  {
    throw std::runtime_error("MatrixLinearStateSolver step size mismatch");
  }

  state = reference;
  for (Index i = 0; i < numStates(); ++i)
  {
    state[i] += step[i];
  }
}

void MatrixLinearStateSolver::initializeReferenceState(
    Vector<Real>& state) const
{
  if (has_reference_state_)
  {
    state = reference_state_;
    return;
  }

  if (state.size() != numStates())
  {
    state.resize(numStates());
  }
  else
  {
    state.setZero();
  }
}

} // namespace eq
} // namespace femx
