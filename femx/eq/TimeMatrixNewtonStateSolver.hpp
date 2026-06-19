#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/TimeMatrixResidualEquation.hpp>
#include <femx/eq/TimeStateSolver.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace eq
{

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
                              system::SystemMatrix&             next_state_jac,
                              system::LinearSolver&             lin_solver);

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
  system::SystemMatrix&              next_state_jac_;
  system::LinearSolver&              lin_solver_;
  TimeMatrixNewtonStateSolverOptions options_;
  Vector<Real>                       init_state_;
  bool                               has_init_state_{false};
};

} // namespace eq
} // namespace femx
