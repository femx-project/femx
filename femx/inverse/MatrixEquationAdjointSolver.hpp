#pragma once

#include <stdexcept>

#include <femx/core/Types.hpp>
#include <femx/system/LinearSolver.hpp>
#include <femx/system/SystemMatrix.hpp>
#include <femx/equation/AssembledResidualEquation.hpp>
#include <femx/inverse/AdjointSolver.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Solves R_u(u,m)^T lambda = rhs using a matrix state Jacobian. */
class MatrixEquationAdjointSolver final : public AdjointSolver
{
public:
  MatrixEquationAdjointSolver(const equation::AssembledResidualEquation& eq,
                              system::SystemMatrix&                 state_jac,
                              system::LinearSolver&           lin_solver)
    : equation_(eq),
      state_jacobian_(state_jac),
      linear_solver_(lin_solver)
  {
    if (equation_.numResiduals() != equation_.numStates())
    {
      throw std::runtime_error(
          "MatrixEquationAdjointSolver requires square state residual dimensions");
    }
  }

  index_type numStates() const override
  {
    return equation_.numStates();
  }

  index_type numParams() const override
  {
    return equation_.numParams();
  }

  index_type numResiduals() const override
  {
    return equation_.numResiduals();
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

    equation_.assembleStateJac(state, params, state_jacobian_);
    state_jacobian_.finalize();
    linear_solver_.solveT(state_jacobian_, rhs, adjoint);
    if (adjoint.size() != numResiduals())
    {
      throw std::runtime_error(
          "MatrixEquationAdjointSolver adjoint size mismatch");
    }
  }

private:
  const equation::AssembledResidualEquation& equation_;
  system::SystemMatrix&                 state_jacobian_;
  system::LinearSolver&           linear_solver_;
};

} // namespace inverse
} // namespace femx
