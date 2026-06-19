#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/MatrixResidualEquation.hpp>
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
                          system::SystemMatrix&         state_jac,
                          system::LinearSolver&         lin_solver);

  void setReferenceState(const Vector<Real>& state);

  void clearReferenceState();

  Index numStates() const override;

  Index numParams() const override;

  void solve(const Vector<Real>& prm, Vector<Real>& state) override;

private:
  void initializeReferenceState(Vector<Real>& state) const;

private:
  const MatrixResidualEquation& eq_;
  system::SystemMatrix&         state_jac_;
  system::LinearSolver&         lin_solver_;
  Vector<Real>                  reference_state_;
  bool                          has_reference_state_{false};
};

} // namespace eq
} // namespace femx
