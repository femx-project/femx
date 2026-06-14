#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/inverse/AdjointSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace inverse
{

/** @brief Solves R_u(u,m)^T lambda = rhs using a matrix state Jacobian. */
class MatrixEquationAdjointSolver final : public AdjointSolver
{
public:
  MatrixEquationAdjointSolver(const eq::AssembledResidualEquation& eq,
                              system::SystemMatrix&                state_jac,
                              system::LinearSolver&                lin_solver)
    : eq_(eq),
      state_jac_(state_jac),
      linear_solver_(lin_solver)
  {
    if (eq_.numRes() != eq_.numStates())
    {
      throw std::runtime_error(
          "MatrixEquationAdjointSolver requires square state residual dimensions");
    }
  }

  Index numStates() const override
  {
    return eq_.numStates();
  }

  Index numParams() const override
  {
    return eq_.numParams();
  }

  Index numRes() const override
  {
    return eq_.numRes();
  }

  void solve(const Vector& state,
             const Vector& params,
             const Vector& rhs,
             Vector&       adjoint) override
  {
    if (state.size() != numStates() || params.size() != numParams()
        || rhs.size() != numStates())
    {
      throw std::runtime_error(
          "MatrixEquationAdjointSolver size mismatch");
    }

    eq_.assembleStateJac(state, params, state_jac_);
    state_jac_.finalize();
    linear_solver_.solveT(state_jac_, rhs, adjoint);
    if (adjoint.size() != numRes())
    {
      throw std::runtime_error(
          "MatrixEquationAdjointSolver adjoint size mismatch");
    }
  }

private:
  const eq::AssembledResidualEquation& eq_;
  system::SystemMatrix&                state_jac_;
  system::LinearSolver&                linear_solver_;
};

} // namespace inverse
} // namespace femx
