#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/MatrixResidualEquation.hpp>
#include <femx/solve/StateSolver.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearSolver.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace solve
{

using problem::MatrixResidualEquation;

/** @brief Solver for residual equations affine in the state.
 *
 * Solves one matrix linearization
 *
 *   R_u(u_ref, m) du = -R(u_ref, m),  u = u_ref + du.
 *
 * For equations affine in u this is the exact state solve.
 */
class MatrixLinearStateSolver final : public StateSolver
{
public:
  MatrixLinearStateSolver(const MatrixResidualEquation& eq,
                          algebra::SystemMatrix&         state_jac,
                          algebra::LinearSolver&         lin_solver);

  void setReferenceState(const Vector<Real>& state);

  void clearReferenceState();

  Index numStates() const override;

  Index numParams() const override;

  void solve(const Vector<Real>& prm, Vector<Real>& state) override;

private:
  void initializeReferenceState(Vector<Real>& state) const;

private:
  const MatrixResidualEquation& eq_;
  algebra::SystemMatrix&         state_jac_;
  algebra::LinearSolver&         lin_solver_;
  Vector<Real>                  reference_state_;
  bool                          has_reference_state_{false};
};

} // namespace solve
} // namespace femx
