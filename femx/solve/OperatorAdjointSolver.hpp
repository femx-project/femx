#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/ResidualEquation.hpp>
#include <femx/solve/AdjointSolver.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearSolver.hpp>

namespace femx
{
namespace solve
{

/** @brief Compatibility solver for R_u(u,m)^T lambda = rhs using an operator. */
class OperatorAdjointSolver final : public AdjointSolver
{
public:
  OperatorAdjointSolver(const problem::ResidualEquation& eq,
                        algebra::LinearSolver&       lin_solver);

  Index numStates() const override;

  Index numParams() const override;

  Index numRes() const override;

  void solve(const Vector<Real>& state,
             const Vector<Real>& prm,
             const Vector<Real>& rhs,
             Vector<Real>&       adjoint) override;

private:
  const problem::ResidualEquation& eq_;
  algebra::LinearSolver&       lin_solver_;
};

} // namespace solve
} // namespace femx
