#pragma once

#include <cmath>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/eq/StateSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace eq
{

struct AssembledNewtonStateSolverOptions
{
  Index max_its        = 20;
  Real  res_tol        = 1.0e-10;
  Real  step_tolerance = 0.0;
};

/** @brief Newton solver for R(u,m)=0 using an assembled state Jacobian. */
class AssembledNewtonStateSolver final : public StateSolver
{
public:
  AssembledNewtonStateSolver(const AssembledResidualEquation& equation,
                             system::SystemMatrix&            state_jac,
                             system::LinearSolver&            lin_solver)
    : eq_(equation),
      state_jac_(state_jac),
      linear_solver_(lin_solver)
  {
    if (eq_.numRes() != eq_.numStates())
    {
      throw std::runtime_error(
          "AssembledNewtonStateSolver requires square state residual dimensions");
    }
  }

  AssembledNewtonStateSolverOptions& options()
  {
    return options_;
  }

  const AssembledNewtonStateSolverOptions& options() const
  {
    return options_;
  }

  void setInitialState(const Vector& state)
  {
    if (state.size() != numStates())
    {
      throw std::runtime_error(
          "AssembledNewtonStateSolver initial state size mismatch");
    }
    initial_state_     = state;
    has_initial_state_ = true;
  }

  void clearInitialState()
  {
    initial_state_     = Vector{};
    has_initial_state_ = false;
  }

  Index numStates() const override
  {
    return eq_.numStates();
  }

  Index numParams() const override
  {
    return eq_.numParams();
  }

  void solve(const Vector& params, Vector& state) override
  {
    if (params.size() != numParams())
    {
      throw std::runtime_error(
          "AssembledNewtonStateSolver parameter size mismatch");
    }

    initializeState(state);

    Vector res;
    Vector rhs;
    Vector step;
    for (Index i = 0; i <= options_.max_its; ++i)
    {
      eq_.res(state, params, res);
      if (res.size() != eq_.numRes())
      {
        throw std::runtime_error(
            "AssembledNewtonStateSolver residual size mismatch");
      }

      if (norm2(res) <= options_.res_tol * options_.res_tol)
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

      eq_.assembleStateJac(state, params, state_jac_);
      state_jac_.finalize();
      linear_solver_.solve(state_jac_, rhs, step);
      if (step.size() != numStates())
      {
        throw std::runtime_error(
            "AssembledNewtonStateSolver step size mismatch");
      }

      for (Index i = 0; i < numStates(); ++i)
      {
        state[i] += step[i];
      }

      if (norm2(step) <= options_.step_tolerance * options_.step_tolerance)
      {
        return;
      }
    }

    throw std::runtime_error("AssembledNewtonStateSolver failed to converge");
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

  static void resize(Vector& out, Index size)
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

  static Real norm2(const Vector& x)
  {
    Real sum = 0.0;
    for (Index i = 0; i < x.size(); ++i)
    {
      sum += x[i] * x[i];
    }
    return sum;
  }

private:
  const AssembledResidualEquation&  eq_;
  system::SystemMatrix&             state_jac_;
  system::LinearSolver&             linear_solver_;
  AssembledNewtonStateSolverOptions options_;
  Vector                            initial_state_;
  bool                              has_initial_state_{false};
};

} // namespace eq
} // namespace femx
