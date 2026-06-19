#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/TimeResidualEquation.hpp>
#include <femx/eq/TimeStateSolver.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>

namespace femx
{
namespace eq
{

struct TimeNewtonStateSolverOptions
{
  Index max_its        = 20;
  Real  res_tol        = 1.0e-10;
  Real  step_tolerance = 0.0;
};

/**
 * @brief Time-marching Newton solver for R_k(u_{k+1}, u_k, m) = 0.
 *
 * The initial state defaults to zero. Each next-state Newton iteration starts
 * from the previous time level, which is a practical default for parabolic and
 * flow time stepping.
 */
class TimeNewtonStateSolver final : public TimeStateSolver
{
public:
  TimeNewtonStateSolver(const TimeResidualEquation& eq,
                        system::LinearSolver&       lin_solver);

  TimeNewtonStateSolverOptions& options();

  const TimeNewtonStateSolverOptions& options() const;

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
  const TimeResidualEquation&  eq_;
  system::LinearSolver&        lin_solver_;
  TimeNewtonStateSolverOptions options_;
  Vector<Real>                 init_state_;
  bool                         has_init_state_{false};
};

} // namespace eq
} // namespace femx
