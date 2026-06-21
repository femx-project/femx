#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/solve/TimeTrajectory.hpp>

namespace femx
{
namespace solve
{

struct TimeStepperOptions
{
  Index max_its            = 20;
  Real  residual_tolerance = 1.0e-10;
  Real  step_tolerance     = 0.0;
};

/** @brief Time-marching Newton solver over problem::TimeResidual. */
class TimeStepper final
{
public:
  TimeStepper(const problem::TimeResidual& problem,
              linalg::LinearSolver&        linear_solver);

  TimeStepperOptions& options();

  const TimeStepperOptions& options() const;

  void setInitialState(const Vector<Real>& state);

  void clearInitialState();

  Index numSteps() const;

  Index numStates() const;

  Index numParams() const;

  Index numResiduals() const;

  void solve(const Vector<Real>& prm,
             TimeTrajectory&     trajectory);

private:
  class NextStateJacobian final : public linalg::LinearOperator
  {
  public:
    explicit NextStateJacobian(const TimeStepper& owner);

    void reset(problem::TimeContext ctx);

    Index numRows() const override;

    Index numCols() const override;

    void apply(const Vector<Real>& dir,
               Vector<Real>&       out) const override;

    void applyT(const Vector<Real>& dir,
                Vector<Real>&       out) const override;

  private:
    const TimeStepper&   owner_;
    problem::TimeContext ctx_;
  };

  void solveStep(Index               step,
                 const Vector<Real>& prm,
                 const Vector<Real>& prev_state,
                 Vector<Real>&       next_state);

  void initializeInitialState(Vector<Real>& state) const;

  static void resize(Vector<Real>& out,
                     Index         size);

  static Real squaredNorm(const Vector<Real>& x);

private:
  const problem::TimeResidual& problem_;
  linalg::LinearSolver&        linear_solver_;
  problem::TimeDims      dims_;
  TimeStepperOptions           options_;
  Vector<Real>                 init_state_;
  bool                         has_init_state_{false};
};

} // namespace solve
} // namespace femx
