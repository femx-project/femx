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

struct MatrixNewtonStateSolverOptions
{
  Index max_its        = 20;
  Real  res_tol        = 1.0e-10;
  Real  step_tolerance = 0.0;
};

/** @brief Newton solver for R(u,m)=0 using a matrix state Jacobian. */
class MatrixNewtonStateSolver final : public StateSolver
{
public:
  MatrixNewtonStateSolver(const MatrixResidualEquation& eq,
                          system::SystemMatrix&         state_jac,
                          system::LinearSolver&         lin_solver);

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
  system::SystemMatrix&          state_jac_;
  system::LinearSolver&          lin_solver_;
  MatrixNewtonStateSolverOptions options_;
  Vector<Real>                   init_state_;
  bool                           has_init_state_{false};
};

} // namespace eq
} // namespace femx
