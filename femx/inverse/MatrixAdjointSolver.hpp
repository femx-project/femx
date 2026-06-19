#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/MatrixResidualEquation.hpp>
#include <femx/inverse/AdjointSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace inverse
{

/** @brief Solves R_u(u,m)^T lambda = rhs using a matrix state Jacobian. */
class MatrixAdjointSolver final : public AdjointSolver
{
public:
  MatrixAdjointSolver(const eq::MatrixResidualEquation& eq,
                      system::SystemMatrix&             state_jac,
                      system::LinearSolver&             lin_solver);

  Index numStates() const override;

  Index numParams() const override;

  Index numRes() const override;

  void solve(const Vector<Real>& state,
             const Vector<Real>& prm,
             const Vector<Real>& rhs,
             Vector<Real>&       adjoint) override;

private:
  const eq::MatrixResidualEquation& eq_;
  system::SystemMatrix&             state_jac_;
  system::LinearSolver&             lin_solver_;
};

} // namespace inverse
} // namespace femx
