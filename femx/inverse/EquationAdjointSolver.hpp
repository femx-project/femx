#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/system/LinearSolver.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/eq/StateJacobianOperator.hpp>
#include <femx/inverse/AdjointSolver.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Solves R_u(u,m)^T lambda = rhs using a LinearSolver. */
class EquationAdjointSolver final : public AdjointSolver
{
public:
  EquationAdjointSolver(const equation::ResidualEquation& equation,
                        system::LinearSolver&     lin_solver)
    : equation_(equation),
      linear_solver_(lin_solver)
  {
    if (equation_.numResiduals() != equation_.numStates())
    {
      throw std::runtime_error(
          "EquationAdjointSolver requires square state residual dimensions");
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
      throw std::runtime_error("EquationAdjointSolver size mismatch");
    }

    const equation::StateJacobianOperator jacobian(equation_, state, params);
    linear_solver_.solveT(jacobian, rhs, adjoint);
    if (adjoint.size() != numResiduals())
    {
      throw std::runtime_error("EquationAdjointSolver adjoint size mismatch");
    }
  }

private:
  const equation::ResidualEquation& equation_;
  system::LinearSolver&     linear_solver_;
};

} // namespace inverse
} // namespace femx
