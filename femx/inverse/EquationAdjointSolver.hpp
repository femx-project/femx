#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/eq/StateJacobianOperator.hpp>
#include <femx/inverse/AdjointSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>

namespace femx
{
namespace inverse
{

/** @brief Solves R_u(u,m)^T lambda = rhs using a LinearSolver. */
class EquationAdjointSolver final : public AdjointSolver
{
public:
  EquationAdjointSolver(const eq::ResidualEquation& equation,
                        system::LinearSolver&       lin_solver)
    : eq_(equation),
      linear_solver_(lin_solver)
  {
    if (eq_.numRes() != eq_.numStates())
    {
      throw std::runtime_error(
          "EquationAdjointSolver requires square state residual dimensions");
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
      throw std::runtime_error("EquationAdjointSolver size mismatch");
    }

    const eq::StateJacobianOperator jac(eq_, state, params);
    linear_solver_.solveT(jac, rhs, adjoint);
    if (adjoint.size() != numRes())
    {
      throw std::runtime_error("EquationAdjointSolver adjoint size mismatch");
    }
  }

private:
  const eq::ResidualEquation& eq_;
  system::LinearSolver&       linear_solver_;
};

} // namespace inverse
} // namespace femx
