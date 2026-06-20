#include <iostream>

#include <femx/problem/TimeResidual.hpp>
#include <femx/solve/TimeStepper.hpp>
#include <femx/algebra/DenseLinearSolver.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class ScalarImplicitTimeResidual final : public problem::TimeResidual
{
public:
  problem::TimeDimensions dimensions() const override
  {
    return {3, 1, 1, 1};
  }

  void residual(const problem::TimeContext& ctx,
                Vector<Real>&               out) const override
  {
    resize(out, 1);
    out[0] = (*ctx.next_state)[0] - (*ctx.previous_state)[0]
             - static_cast<Real>(ctx.step + 1) * (*ctx.prm)[0];
  }

  void applyJacobian(const problem::TimeContext& ctx,
                     problem::VariableBlock     wrt,
                     const Vector<Real>&        dir,
                     Vector<Real>&              out) const override
  {
    resize(out, 1);
    switch (wrt)
    {
    case problem::VariableBlock::PreviousState:
      out[0] = -dir[0];
      return;

    case problem::VariableBlock::NextState:
      out[0] = dir[0];
      return;

    case problem::VariableBlock::Parameter:
      out[0] = -static_cast<Real>(ctx.step + 1) * dir[0];
      return;
    }
  }

  void applyJacobianT(const problem::TimeContext& ctx,
                      problem::VariableBlock     wrt,
                      const Vector<Real>&        adjoint,
                      Vector<Real>&              out) const override
  {
    resize(out, 1);
    switch (wrt)
    {
    case problem::VariableBlock::PreviousState:
      out[0] = -adjoint[0];
      return;

    case problem::VariableBlock::NextState:
      out[0] = adjoint[0];
      return;

    case problem::VariableBlock::Parameter:
      out[0] = -static_cast<Real>(ctx.step + 1) * adjoint[0];
      return;
    }
  }

private:
  static void resize(Vector<Real>& out, Index size)
  {
    if (out.size() != size)
    {
      out.resize(size);
    }
    else
    {
      out.setZero();
    }
  }
};

class TimeStepperTests : public TestBase
{
public:
  TestOutcome solvesImplicitSteps()
  {
    TestStatus status;
    status = true;

    ScalarImplicitTimeResidual problem;
    algebra::DenseLinearSolver  linear_solver;
    solve::TimeStepper         stepper(problem, linear_solver);

    status *= (stepper.numSteps() == 3);
    status *= (stepper.numStates() == 1);
    status *= (stepper.numParams() == 1);
    status *= (stepper.numResiduals() == 1);

    Vector<Real> initial(1);
    initial[0] = 1.0;
    stepper.setInitialState(initial);

    Vector<Real> prm(1);
    prm[0] = 2.0;

    solve::TimeTrajectory trajectory;
    stepper.solve(prm, trajectory);

    status *= (trajectory.numSteps() == 3);
    status *= (trajectory.numTimeLevels() == 4);
    status *= isEqual(trajectory[0][0], 1.0);
    status *= isEqual(trajectory[1][0], 3.0);
    status *= isEqual(trajectory[2][0], 7.0);
    status *= isEqual(trajectory[3][0], 13.0);

    return status.report(__func__);
  }

  TestOutcome defaultsInitialStateToZero()
  {
    TestStatus status;
    status = true;

    ScalarImplicitTimeResidual problem;
    algebra::DenseLinearSolver  linear_solver;
    solve::TimeStepper         stepper(problem, linear_solver);

    Vector<Real> prm(1);
    prm[0] = 1.5;

    solve::TimeTrajectory trajectory;
    stepper.solve(prm, trajectory);

    status *= isEqual(trajectory[0][0], 0.0);
    status *= isEqual(trajectory[1][0], 1.5);
    status *= isEqual(trajectory[2][0], 4.5);
    status *= isEqual(trajectory[3][0], 9.0);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running time stepper tests:\n";

  femx::tests::TimeStepperTests test;

  femx::tests::TestingResults result;
  result += test.solvesImplicitSteps();
  result += test.defaultsInitialStateToZero();

  return result.summary();
}
