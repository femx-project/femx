#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/inverse/AdjointSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>

namespace femx
{
namespace inverse
{

/** @brief Solves R_u(u,m)^T lambda = rhs using a StateJacobianOperator. */
class OperatorAdjointSolver final : public AdjointSolver
{
public:
  OperatorAdjointSolver(const eq::ResidualEquation& eq,
                        system::LinearSolver&       lin_solver);

  Index numStates() const override;

  Index numParams() const override;

  Index numRes() const override;

  void solve(const Vector<Real>& state,
             const Vector<Real>& prm,
             const Vector<Real>& rhs,
             Vector<Real>&       adjoint) override;

private:
  const eq::ResidualEquation& eq_;
  system::LinearSolver&       lin_solver_;
};

} // namespace inverse
} // namespace femx
