#pragma once

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

/** @brief Solver for residual equations affine in the state.
 *
 * Solves one assembled linearization
 *
 *   R_u(u_ref, m) du = -R(u_ref, m),  u = u_ref + du.
 *
 * For equations affine in u this is the exact state solve.
 */
class AssembledLinearStateSolver final : public StateSolver
{
public:
  AssembledLinearStateSolver(const AssembledResidualEquation& equation,
                             system::SystemMatrix&            state_jac,
                             system::LinearSolver&            lin_solver)
    : eq_(equation),
      state_jac_(state_jac),
      linear_solver_(lin_solver)
  {
    if (eq_.numRes() != eq_.numStates())
    {
      throw std::runtime_error(
          "AssembledLinearStateSolver requires square state residual dimensions");
    }
  }

  void setReferenceState(const Vector& state)
  {
    if (state.size() != numStates())
    {
      throw std::runtime_error(
          "AssembledLinearStateSolver reference state size mismatch");
    }
    reference_state_     = state;
    has_reference_state_ = true;
  }

  void clearReferenceState()
  {
    reference_state_     = Vector{};
    has_reference_state_ = false;
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
          "AssembledLinearStateSolver parameter size mismatch");
    }

    Vector reference;
    initializeReferenceState(reference);

    Vector rhs;
    eq_.res(reference, params, rhs);
    if (rhs.size() != eq_.numRes())
    {
      throw std::runtime_error(
          "AssembledLinearStateSolver residual size mismatch");
    }

    for (Index i = 0; i < rhs.size(); ++i)
    {
      rhs[i] = -rhs[i];
    }

    eq_.assembleStateJac(reference, params, state_jac_);
    state_jac_.finalize();

    Vector step;
    linear_solver_.solve(state_jac_, rhs, step);
    if (step.size() != numStates())
    {
      throw std::runtime_error(
          "AssembledLinearStateSolver step size mismatch");
    }

    state = reference;
    for (Index i = 0; i < numStates(); ++i)
    {
      state[i] += step[i];
    }
  }

private:
  void initializeReferenceState(Vector& state) const
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

private:
  const AssembledResidualEquation& eq_;
  system::SystemMatrix&            state_jac_;
  system::LinearSolver&            linear_solver_;
  Vector                           reference_state_;
  bool                             has_reference_state_{false};
};

} // namespace eq
} // namespace femx
