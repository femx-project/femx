#pragma once

#include <cmath>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/system/LinearSolver.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/eq/StateJacobianOperator.hpp>
#include <femx/eq/StateSolver.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace equation
{

struct NewtonStateSolverOptions
{
  index_type max_its     = 20;
  real_type  residual_tolerance = 1.0e-10;
  real_type  step_tolerance     = 0.0;
};

/** @brief Newton solver for R(u,m)=0 using a matrix-free state Jacobian. */
class NewtonStateSolver final : public StateSolver
{
public:
  NewtonStateSolver(const ResidualEquation& equation,
                    system::LinearSolver&     lin_solver)
    : equation_(equation),
      linear_solver_(lin_solver)
  {
    if (equation_.numResiduals() != equation_.numStates())
    {
      throw std::runtime_error(
          "NewtonStateSolver requires square state residual dimensions");
    }
  }

  NewtonStateSolverOptions& options()
  {
    return options_;
  }

  const NewtonStateSolverOptions& options() const
  {
    return options_;
  }

  void setInitialState(const Vector& state)
  {
    if (state.size() != numStates())
    {
      throw std::runtime_error(
          "NewtonStateSolver initial state size mismatch");
    }
    initial_state_     = state;
    has_initial_state_ = true;
  }

  void clearInitialState()
  {
    initial_state_     = Vector{};
    has_initial_state_ = false;
  }

  index_type numStates() const override
  {
    return equation_.numStates();
  }

  index_type numParams() const override
  {
    return equation_.numParams();
  }

  void solve(const Vector& params, Vector& state) override
  {
    if (params.size() != numParams())
    {
      throw std::runtime_error("NewtonStateSolver parameter size mismatch");
    }

    initializeState(state);

    Vector residual;
    Vector rhs;
    Vector step;
    for (index_type iteration = 0; iteration <= options_.max_its; ++iteration)
    {
      equation_.residual(state, params, residual);
      if (residual.size() != equation_.numResiduals())
      {
        throw std::runtime_error("NewtonStateSolver residual size mismatch");
      }

      if (norm2(residual) <= options_.residual_tolerance * options_.residual_tolerance)
      {
        return;
      }
      if (iteration == options_.max_its)
      {
        break;
      }

      resize(rhs, residual.size());
      for (index_type i = 0; i < residual.size(); ++i)
      {
        rhs[i] = -residual[i];
      }

      const StateJacobianOperator jacobian(equation_, state, params);
      linear_solver_.solve(jacobian, rhs, step);
      if (step.size() != numStates())
      {
        throw std::runtime_error("NewtonStateSolver step size mismatch");
      }

      for (index_type i = 0; i < numStates(); ++i)
      {
        state[i] += step[i];
      }

      if (norm2(step) <= options_.step_tolerance * options_.step_tolerance)
      {
        return;
      }
    }

    throw std::runtime_error("NewtonStateSolver failed to converge");
  }

private:
  void initializeState(Vector& state) const
  {
    if (has_initial_state_)
    {
      state = initial_state_;
      return;
    }
    resize(state, numStates());
  }

  static void resize(Vector& out, index_type size)
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

  static real_type norm2(const Vector& x)
  {
    real_type sum = 0.0;
    for (index_type i = 0; i < x.size(); ++i)
    {
      sum += x[i] * x[i];
    }
    return sum;
  }

private:
  const ResidualEquation&  equation_;
  system::LinearSolver&      linear_solver_;
  NewtonStateSolverOptions options_;
  Vector                   initial_state_;
  bool                     has_initial_state_{false};
};

} // namespace equation
} // namespace femx
