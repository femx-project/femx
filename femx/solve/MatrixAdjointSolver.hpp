#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/MatrixResidualEquation.hpp>
#include <femx/solve/AdjointSolver.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearSolver.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace solve
{

/** @brief Compatibility solver for R_u(u,m)^T lambda = rhs using a matrix. */
class MatrixAdjointSolver final : public AdjointSolver
{
public:
  MatrixAdjointSolver(const problem::MatrixResidualEquation& eq,
                      algebra::SystemMatrix&             state_jac,
                      algebra::LinearSolver&             lin_solver);

  Index numStates() const override;

  Index numParams() const override;

  Index numRes() const override;

  void solve(const Vector<Real>& state,
             const Vector<Real>& prm,
             const Vector<Real>& rhs,
             Vector<Real>&       adjoint) override;

private:
  const problem::MatrixResidualEquation& eq_;
  algebra::SystemMatrix&             state_jac_;
  algebra::LinearSolver&             lin_solver_;
};

} // namespace solve
} // namespace femx
