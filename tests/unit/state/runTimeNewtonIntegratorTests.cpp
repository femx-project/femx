#include <cmath>
#include <vector>

#include <femx/linalg/native/DenseLinearSolver.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/state/TimeNewtonIntegrator.hpp>
#include <femx/state/TimeStateMonitor.hpp>
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
    throw std::runtime_error("TimeNewtonIntegrator test expected a scalar");
  }
  return (*value)[0];
}

Real scalar(VectorView<const Real> value)
{
  if (value.size() != 1)
  {
    throw std::runtime_error("TimeNewtonIntegrator test expected a scalar");
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
      throw std::runtime_error("TimeNewtonIntegrator test step out of range");
    }
    if (ctx.prm == nullptr || ctx.prm->size() != 1)
    {
      throw std::runtime_error(
          "TimeNewtonIntegrator test parameter size mismatch");
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

class RecordingTimeMonitor final : public state::TimeStateMonitor
{
public:
  void observe(Index               level,
               const Vector<Real>& state) override
  {
    levels.push_back(level);
    values.push_back(scalar(VectorView<const Real>(state.data(), state.size())));
  }

  std::vector<Index> levels;
  std::vector<Real>  values;
};

class RecordingStepMonitor final : public state::TimeStateMonitor
{
public:
  void observe(Index               level,
               const Vector<Real>& state) override
  {
    initial_levels.push_back(level);
    initial_values.push_back(
        scalar(VectorView<const Real>(state.data(), state.size())));
  }

  bool observeStep(const state::TimeStepStateContext& ctx) override
  {
    step_levels.push_back(ctx.level);
    totals.push_back(ctx.total_steps);
    previous_values.push_back(
        scalar(VectorView<const Real>(ctx.previous.data(), ctx.previous.size())));
    current_values.push_back(
        scalar(VectorView<const Real>(ctx.current.data(), ctx.current.size())));
    assembly_seconds.push_back(ctx.assembly_seconds);
    solve_seconds.push_back(ctx.solve_seconds);
    return false;
  }

  std::vector<Index> initial_levels;
  std::vector<Real>  initial_values;
  std::vector<Index> step_levels;
  std::vector<Index> totals;
  std::vector<Real>  previous_values;
  std::vector<Real>  current_values;
  std::vector<Real>  assembly_seconds;
  std::vector<Real>  solve_seconds;
};

} // namespace

class TimeNewtonIntegratorTests
{
public:
  TestOutcome solvesNonlinearStepEquations()
  {
    TestStatus status;
    status = true;

    constexpr Index              steps = 4;
    NonlinearScalarTimeResidual  problem(steps);
    linalg::DenseLinearSolver    lin_solver;
    state::TimeNewtonIntegrator solver(problem, lin_solver);
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

    RecordingTimeMonitor monitor;
    solver.setMonitor(&monitor);
    solver.solve(prm);
    solver.clearMonitor();

    status *= static_cast<Index>(monitor.levels.size()) == steps + 1;
    status *= static_cast<Index>(monitor.values.size()) == steps + 1;
    for (Index level = 0; level < static_cast<Index>(monitor.values.size()); ++level)
    {
      status *= monitor.levels[level] == level;
      status *= near(monitor.values[level], expected[level], 1.0e-11);
    }

    return status.report(__func__);
  }

  TestOutcome stepMonitorReceivesStepContext()
  {
    TestStatus status;
    status = true;

    constexpr Index              steps = 3;
    NonlinearScalarTimeResidual  problem(steps);
    linalg::DenseLinearSolver    lin_solver;
    state::TimeNewtonIntegrator solver(problem, lin_solver);
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

    RecordingStepMonitor monitor;
    state::TimeTrajectory tr;
    solver.setMonitor(&monitor);
    solver.solve(prm, tr);
    solver.clearMonitor();

    status *= static_cast<Index>(monitor.initial_levels.size()) == 1;
    status *= monitor.initial_levels.front() == 0;
    status *= near(monitor.initial_values.front(), expected[0], 1.0e-11);
    status *= static_cast<Index>(monitor.step_levels.size()) == steps;
    status *= static_cast<Index>(monitor.previous_values.size()) == steps;
    status *= static_cast<Index>(monitor.current_values.size()) == steps;
    for (Index step = 0; step < steps; ++step)
    {
      status *= monitor.step_levels[step] == step + 1;
      status *= monitor.totals[step] == steps;
      status *= near(monitor.previous_values[step], expected[step], 1.0e-11);
      status *= near(monitor.current_values[step], expected[step + 1], 1.0e-11);
      status *= near(tr[step + 1][0], expected[step + 1], 1.0e-11);
      status *= monitor.assembly_seconds[step] == 0.0;
      status *= monitor.solve_seconds[step] == 0.0;
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults             results;
  femx::tests::TimeNewtonIntegratorTests test;

  results += test.solvesNonlinearStepEquations();
  results += test.stepMonitorReceivesStepContext();

  return results.summary();
}
