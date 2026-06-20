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

struct MatrixNewtonStateSolverOptions
{
  Index max_its        = 20;
  Real  res_tol        = 1.0e-10;
  Real  step_tolerance = 0.0;
};

/** @brief Compatibility solver. Prefer solve::Newton for new code. */
class MatrixNewtonStateSolver final : public StateSolver
{
public:
  MatrixNewtonStateSolver(const MatrixResidualEquation& eq,
                          algebra::SystemMatrix&         state_jac,
                          algebra::LinearSolver&         lin_solver);

  MatrixNewtonStateSolverOptions& options();

  const MatrixNewtonStateSolverOptions& options() const;

  void setInitialState(const Vector<Real>& state);

  void clearInitialState();

  Index numStates() const override;

  Index numParams() const override;

  void solve(const Vector<Real>& prm, Vector<Real>& state) override;

private:
  void initializeState(Vector<Real>& state) const;

  static void resize(Vector<Real>& out, Index size);

private:
  const MatrixResidualEquation&  eq_;
  algebra::SystemMatrix&          state_jac_;
  algebra::LinearSolver&          lin_solver_;
  MatrixNewtonStateSolverOptions options_;
  Vector<Real>                   init_state_;
  bool                           has_init_state_{false};
};

} // namespace solve
} // namespace femx
