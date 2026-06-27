#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace state
{

struct TimeNewtonOptions
{
  Index max_its            = 20;
  Real  residual_tolerance = 1.0e-10;
  Real  step_tolerance     = 0.0;
};

/** @brief Time integrator using Newton iterations at each time step. */
class TimeNewtonIntegrator final : public TimeIntegrator
{
public:
  TimeNewtonIntegrator(const problem::TimeResidual& problem,
                       linalg::LinearSolver&        lin_solver);

  TimeNewtonOptions& opts();

  const TimeNewtonOptions& opts() const;

  void setInitialState(const Vector<Real>& state);

  void clearInitialState();

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  Index numResiduals() const;

  void solve(const Vector<Real>& prm,
             TimeTrajectory&     tr) override;

  void solve(const Vector<Real>& prm);

private:
  class NextStateJacobian final : public linalg::LinearOperator
  {
  public:
    explicit NextStateJacobian(const TimeNewtonIntegrator& owner);

    void reset(problem::TimeContext ctx);

    Index numRows() const override;

    Index numCols() const override;

    void apply(const Vector<Real>& dir,
               Vector<Real>&       out) const override;

    void applyT(const Vector<Real>& dir,
                Vector<Real>&       out) const override;

  private:
    const TimeNewtonIntegrator& owner_;
    problem::TimeContext         ctx_;
  };

  void solveImpl(const Vector<Real>& prm,
                 TimeTrajectory*     tr);

  void solveStep(Index               step,
                 const Vector<Real>& prm,
                 const Vector<Real>& hist,
                 Vector<Real>&       nxt);

  void initializeInitialState(Vector<Real>& state) const;

  static Real squaredNorm(const Vector<Real>& x);

private:
  const problem::TimeResidual& problem_;
  linalg::LinearSolver&        lin_solver_;
  problem::TimeDims            dims_;
  TimeNewtonOptions            opts_;
  Vector<Real>                 init_state_;
  bool                         has_init_state_{false};
};

} // namespace state
} // namespace femx
