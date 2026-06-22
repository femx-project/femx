#include <cmath>

#include <femx/linalg/DenseLinearSolver.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/state/TimeNewtonStateSolver.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{
namespace
{

Real scalar(const Vector<Real>* value)
{
  if (value == nullptr || value->size() != 1)
  {
    throw std::runtime_error("TimeNewtonStateSolver test expected a scalar");
  }
  return (*value)[0];
}

Real scalar(VectorView<const Real> value)
{
  if (value.size() != 1)
  {
    throw std::runtime_error("TimeNewtonStateSolver test expected a scalar");
  }
  return value[0];
}

class NonlinearScalarTimeResidual final : public problem::TimeResidual
{
public:
  explicit NonlinearScalarTimeResidual(Index steps)
    : steps_(steps)
  {
  }

  problem::TimeDims dims() const override
  {
    return {steps_, 1, 1, 1, 1};
  }

  void res(const problem::TimeContext& ctx,
           Vector<Real>&               out) const override
  {
    checkContext(ctx);
    resizeOrZero(out, 1);
    const Real current = scalar(ctx.historyView().state(0));
    const Real next    = scalar(ctx.nxt);
    out[0]             = next * next - current - (*ctx.prm)[0];
  }

  void applyJac(const problem::TimeContext& ctx,
                problem::VariableBlock      wrt,
                const Vector<Real>&         dir,
                Vector<Real>&               out) const override
  {
    checkContext(ctx);
    resizeOrZero(out, 1);
    out[0] = derivative(ctx, wrt) * dir[0];
  }

  void applyJacT(const problem::TimeContext& ctx,
                 problem::VariableBlock      wrt,
                 const Vector<Real>&         adj,
                 Vector<Real>&               out) const override
  {
    checkContext(ctx);
    resizeOrZero(out, 1);
    out[0] = derivative(ctx, wrt) * adj[0];
  }

private:
  void checkContext(const problem::TimeContext& ctx) const
  {
    if (ctx.step < 0 || ctx.step >= steps_)
    {
      throw std::runtime_error("TimeNewtonStateSolver test step out of range");
    }
    if (ctx.prm == nullptr || ctx.prm->size() != 1)
    {
      throw std::runtime_error(
          "TimeNewtonStateSolver test parameter size mismatch");
    }
    (void) scalar(ctx.historyView().state(0));
    (void) scalar(ctx.nxt);
  }

  Real derivative(const problem::TimeContext& ctx,
                  problem::VariableBlock      wrt) const
  {
    if (wrt.isNextState())
    {
      return 2.0 * scalar(ctx.nxt);
    }
    if (wrt.isParam())
    {
      return -1.0;
    }
    return wrt.historyLag() == 0 ? -1.0 : 0.0;
  }

private:
  Index steps_{0};
};

bool near(Real actual, Real expected, Real tol)
{
  return std::abs(actual - expected) <= tol;
}

} // namespace

class TimeNewtonStateSolverTests
{
public:
  TestOutcome solvesNonlinearStepEquations()
  {
    TestStatus status;
    status = true;

    constexpr Index              steps = 4;
    NonlinearScalarTimeResidual  problem(steps);
    linalg::DenseLinearSolver    lin_solver;
    state::TimeNewtonStateSolver solver(problem, lin_solver);
    solver.opts().residual_tolerance = 1.0e-13;

    Vector<Real> initial(1);
    initial[0] = 1.0;
    solver.setInitialState(initial);

    Vector<Real> prm(1);
    prm[0] = 0.21;

    Vector<Real> expected(steps + 1);
    expected[0] = initial[0];
    for (Index step = 0; step < steps; ++step)
    {
      expected[step + 1] = std::sqrt(expected[step] + prm[0]);
    }

    state::TimeTrajectory tr;
    solver.solve(prm, tr);
    status *= tr.numTimeLevels() == steps + 1;
    for (Index level = 0; level < tr.numTimeLevels(); ++level)
    {
      status *= near(tr[level][0], expected[level], 1.0e-11);
    }

    Vector<Real> observed;
    solver.solve(
        prm,
        [&](Index level, const Vector<Real>& state)
        {
          status *= level == observed.size();
          status *= state.size() == 1;
          observed.push_back(state[0]);
        });
    status *= observed.size() == steps + 1;
    for (Index level = 0; level < observed.size(); ++level)
    {
      status *= near(observed[level], expected[level], 1.0e-11);
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults             results;
  femx::tests::TimeNewtonStateSolverTests test;

  results += test.solvesNonlinearStepEquations();

  return results.summary();
}
