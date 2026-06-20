#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/TimeMatrixResidualEquation.hpp>
#include <femx/solve/TimeStateSolver.hpp>
#include <femx/solve/TimeStateTrajectory.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearSolver.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace solve
{

using problem::TimeMatrixResidualEquation;

struct TimeMatrixNewtonStateSolverOptions
{
  Index max_its        = 20;
  Real  res_tol        = 1.0e-10;
  Real  step_tolerance = 0.0;
};

/** @brief Time-marching Newton solver using assembled next-state Jacobians. */
class TimeMatrixNewtonStateSolver final : public TimeStateSolver
{
public:
  TimeMatrixNewtonStateSolver(const TimeMatrixResidualEquation& eq,
                              algebra::SystemMatrix&             next_state_jac,
                              algebra::LinearSolver&             lin_solver);

  TimeMatrixNewtonStateSolverOptions& options();

  const TimeMatrixNewtonStateSolverOptions& options() const;

  void setInitialState(const Vector<Real>& state);

  void clearInitialState();

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  void solve(const Vector<Real>&  prm,
             TimeStateTrajectory& tr) override;

private:
  void solveStep(Index               step,
                 const Vector<Real>& prm,
                 const Vector<Real>& x,
                 Vector<Real>&       x_next);

  void initializeInitialState(Vector<Real>& state) const;

  static void resize(Vector<Real>& out,
                     Index         size);

private:
  const TimeMatrixResidualEquation&  eq_;
  algebra::SystemMatrix&              next_state_jac_;
  algebra::LinearSolver&              lin_solver_;
  TimeMatrixNewtonStateSolverOptions options_;
  Vector<Real>                       init_state_;
  bool                               has_init_state_{false};
};

} // namespace solve
} // namespace femx
