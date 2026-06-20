#include <stdexcept>

#include <femx/solve/Newton.hpp>

namespace femx
{
namespace solve
{

Newton::Newton(const problem::Residual& problem,
               problem::Linearization&  linearization,
               algebra::LinearSolver&   linear_solver)
  : problem_(problem),
    linearization_(linearization),
    linear_solver_(linear_solver),
    dims_(problem.dimensions())
{
  if (dims_.num_residuals != dims_.num_states)
  {
    throw std::runtime_error(
        "Newton requires square state residual dimensions");
  }
}

NewtonOptions& Newton::options()
{
  return options_;
}

const NewtonOptions& Newton::options() const
{
  return options_;
}

void Newton::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error("Newton initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void Newton::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

Index Newton::numStates() const
{
  return dims_.num_states;
}

Index Newton::numParams() const
{
  return dims_.num_params;
}

Index Newton::numResiduals() const
{
  return dims_.num_residuals;
}

problem::Linearization& Newton::linearization()
{
  return linearization_;
}

const problem::Linearization& Newton::linearization() const
{
  return linearization_;
}

void Newton::solve(const Vector<Real>& prm,
                   Vector<Real>&       state)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error("Newton parameter size mismatch");
  }

  initializeState(state);

  Vector<Real> res;
  Vector<Real> rhs;
  Vector<Real> step;
  for (Index i = 0; i <= options_.max_its; ++i)
  {
    problem_.residual(state, prm, res);
    if (res.size() != numResiduals())
    {
      throw std::runtime_error("Newton residual size mismatch");
    }

    if (squaredNorm(res)
        <= options_.residual_tolerance * options_.residual_tolerance)
    {
      return;
    }
    if (i == options_.max_its)
    {
      break;
    }

    resize(rhs, res.size());
    for (Index k = 0; k < res.size(); ++k)
    {
      rhs[k] = -res[k];
    }

    problem_.linearize(state, prm, linearization_);
    linear_solver_.solve(linearization_.stateJacobian(), rhs, step);
    if (step.size() != numStates())
    {
      throw std::runtime_error("Newton step size mismatch");
    }

    for (Index k = 0; k < numStates(); ++k)
    {
      state[k] += step[k];
    }

    if (squaredNorm(step) <= options_.step_tolerance * options_.step_tolerance)
    {
      return;
    }
  }

  throw std::runtime_error("Newton failed to converge");
}

void Newton::initializeState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resize(state, numStates());
}

void Newton::resize(Vector<Real>& out,
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

Real Newton::squaredNorm(const Vector<Real>& x)
{
  Real value = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    value += x[i] * x[i];
  }
  return value;
}

} // namespace solve
} // namespace femx
