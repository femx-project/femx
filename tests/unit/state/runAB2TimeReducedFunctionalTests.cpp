#include <cmath>
#include <iostream>
#include <vector>

#include <femx/linalg/native/DenseLinearSolver.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/native/DenseMatrixOperator.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/state/EnsembleReducedFunctional.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>
#include <femx/state/TimeReducedFunctional.hpp>
#include <femx/state/TimeStateMonitor.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

namespace
{

Real scalar(VectorView<const Real> value)
{
  if (value.size() != 1)
  {
    throw std::runtime_error("AB2 test received invalid scalar state");
  }
  return value[0];
}

Real scalar(const Vector<Real>* value)
{
  if (value == nullptr)
  {
    throw std::runtime_error("AB2 test received invalid scalar state");
  }
  return scalar(VectorView<const Real>(value->data(), value->size()));
}

class AB2ScalarResidual final : public problem::TimeResidual
{
public:
  AB2ScalarResidual(Index steps, Real dt, Real coeff)
    : steps_(steps),
      dt_(dt),
      coeff_(coeff)
  {
  }

  problem::TimeDims dims() const override
  {
    return {steps_, 1, 1, 1, 2};
  }

  void res(const problem::TimeContext& ctx,
           Vector<Real>&               out) const override
  {
    checkContext(ctx);
    resizeOrZero(out, 1);

    const problem::TimeHistoryView hist    = ctx.historyView();
    const Real                     current = scalar(hist.state(0));
    const Real                     adv =
        ctx.step == 0
                                ? current
                                : 1.5 * current - 0.5 * scalar(hist.state(1));
    out[0] = scalar(ctx.nxt) - current
             - dt_ * (coeff_ * adv + (*ctx.prm)[0]);
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

  bool assembleJac(const problem::TimeContext& ctx,
                   problem::VariableBlock      wrt,
                   linalg::MatrixBuilder&      out) const override
  {
    checkContext(ctx);
    out.resize(1, 1);
    out.setZero();
    out.set(0, 0, derivative(ctx, wrt));
    return true;
  }

private:
  void checkContext(const problem::TimeContext& ctx) const
  {
    if (ctx.step < 0 || ctx.step >= steps_)
    {
      throw std::runtime_error("AB2 test step out of range");
    }
    const problem::TimeHistoryView hist = ctx.historyView();
    (void) scalar(hist.state(0));
    (void) scalar(hist.state(1));
    (void) scalar(ctx.nxt);
    if (ctx.prm == nullptr || ctx.prm->size() != 1)
    {
      throw std::runtime_error("AB2 test parameter size mismatch");
    }
  }

  Real derivative(const problem::TimeContext& ctx,
                  problem::VariableBlock      wrt) const
  {
    if (wrt.isNextState())
    {
      return 1.0;
    }
    if (wrt.isParam())
    {
      return -dt_;
    }
    const Index lag = wrt.historyLag();
    if (lag == 0)
    {
      return ctx.step == 0 ? -1.0 - dt_ * coeff_
                           : -1.0 - 1.5 * dt_ * coeff_;
    }
    if (lag == 1)
    {
      return ctx.step == 0 ? 0.0 : 0.5 * dt_ * coeff_;
    }
    throw std::runtime_error("AB2 test history lag out of range");
  }

private:
  Index steps_{0};
  Real  dt_{0.0};
  Real  coeff_{0.0};
};

class TerminalTrackingObjective final : public problem::TimeObjective
{
public:
  TerminalTrackingObjective(Index steps, Real target)
    : steps_(steps),
      target_(target)
  {
  }

  Index numSteps() const override
  {
    return steps_;
  }

  Index numStates() const override
  {
    return 1;
  }

  Index numParams() const override
  {
    return 1;
  }

  Real value(const state::TimeTrajectory& tr,
             const Vector<Real>&) const override
  {
    const Real diff = tr[steps_][0] - target_;
    return 0.5 * diff * diff;
  }

  void stateGrad(Index                        level,
                 const state::TimeTrajectory& tr,
                 const Vector<Real>&,
                 Vector<Real>& out) const override
  {
    resizeOrZero(out, 1);
    if (level == steps_)
    {
      out[0] = tr[steps_][0] - target_;
    }
  }

  void paramGrad(const state::TimeTrajectory&,
                 const Vector<Real>&,
                 Vector<Real>& out) const override
  {
    resizeOrZero(out, 1);
  }

private:
  Index steps_{0};
  Real  target_{0.0};
};

bool near(Real actual, Real exp, Real tol)
{
  return std::abs(actual - exp) <= tol;
}

class RecordingTimeMonitor final : public state::TimeStateMonitor
{
public:
  void start(Index num_steps,
             Index num_states) override
  {
    ++starts;
    steps  = num_steps;
    states = num_states;
  }

  void observe(Index               level,
               const Vector<Real>& state) override
  {
    levels.push_back(level);
    values.push_back(scalar(VectorView<const Real>(state.data(), state.size())));
  }

  void stop() override
  {
    ++stops;
  }

  Index              starts{0};
  Index              stops{0};
  Index              steps{0};
  Index              states{0};
  std::vector<Index> levels;
  std::vector<Real>  values;
};

} // namespace

class AB2TimeReducedFunctionalTests
{
public:
  TestOutcome monitorSolveMatchesExpectedHistory()
  {
    TestStatus status;
    status = true;

    const Index steps = 5;
    const Real  dt    = 0.2;
    const Real  coeff = 0.7;

    AB2ScalarResidual            problem(steps, dt, coeff);
    linalg::DenseMatrixOperator  J_next;
    linalg::DenseLinearSolver    solver;
    state::TimeLinearIntegrator integrator(problem, J_next, solver);

    Vector<Real> initial(1);
    initial[0] = 0.35;
    integrator.setInitialState(initial);

    Vector<Real> prm(1);
    prm[0] = 0.4;

    Vector<Real> expected(steps + 1);
    expected[0] = initial[0];
    for (Index step = 0; step < steps; ++step)
    {
      const Real current = expected[step];
      const Real old     = step == 0 ? expected[0] : expected[step - 1];
      const Real adv     = step == 0 ? current : 1.5 * current - 0.5 * old;
      expected[step + 1] = current + dt * (coeff * adv + prm[0]);
    }

    RecordingTimeMonitor monitor;
    integrator.setMonitor(&monitor);
    integrator.solve(prm);
    integrator.clearMonitor();

    status *= monitor.starts == 1;
    status *= monitor.stops == 1;
    status *= monitor.steps == steps;
    status *= monitor.states == 1;
    status *= static_cast<Index>(monitor.levels.size()) == steps + 1;
    status *= static_cast<Index>(monitor.values.size()) == steps + 1;
    for (Index level = 0; level < static_cast<Index>(monitor.levels.size()); ++level)
    {
      status *= monitor.levels[level] == level;
      status *= near(monitor.values[level], expected[level], 1.0e-12);
    }

    state::TimeTrajectory tr;
    integrator.solve(prm, tr);

    status *= tr.numTimeLevels() == steps + 1;
    for (Index level = 0; level < tr.numTimeLevels(); ++level)
    {
      status *= near(tr[level][0], expected[level], 1.0e-12);
    }

    return status.report(__func__);
  }

  TestOutcome gradientMatchesFiniteDifferenceAndDescends()
  {
    TestStatus status;
    status = true;

    AB2ScalarResidual                problem(4, 0.25, 0.7);
    linalg::DenseMatrixOperator      J_next;
    linalg::DenseMatrixOperator      J_hist;
    linalg::DenseLinearSolver        solver;
    state::TimeLinearIntegrator     integrator(problem, J_next, solver);
    femx::problem::TimeLinearization adj_lin;

    Vector<Real> initial(1);
    initial[0] = 0.35;
    integrator.setInitialState(initial);

    TerminalTrackingObjective    obj(problem.dims().nt, 1.1);
    state::TimeReducedFunctional reduced(
        integrator, problem, adj_lin, J_next, J_hist, solver, obj);

    Vector<Real> prm(1);
    prm[0] = 0.4;

    Vector<Real> grad;
    const Real   value  = reduced.valueGrad(prm, grad);
    status             *= grad.size() == 1;

    const Real   eps    = 1.0e-6;
    Vector<Real> plus   = prm;
    Vector<Real> minus  = prm;
    plus[0]            += eps;
    minus[0]           -= eps;
    const Real fd       = (reduced.value(plus) - reduced.value(minus)) / (2.0 * eps);
    status             *= near(grad[0], fd, 1.0e-7);

    Vector<Real> trial       = prm;
    Real         trial_value = value;
    Real         alpha       = 1.0;
    bool         descended   = false;
    for (Index i = 0; i < 20; ++i)
    {
      trial[0]    = prm[0] - alpha * grad[0];
      trial_value = reduced.value(trial);
      if (trial_value < value)
      {
        descended = true;
        break;
      }
      alpha *= 0.5;
    }
    status *= descended;

    std::cout << "AB2 reduced functional: value " << value
              << " -> " << trial_value
              << ", grad = " << grad[0]
              << ", fd = " << fd
              << ", alpha = " << alpha << '\n';

    return status.report(__func__);
  }

  TestOutcome ensembleReducedFunctionalWrapsTimeReducedFunctional()
  {
    TestStatus status;
    status = true;

    AB2ScalarResidual                problem(4, 0.25, 0.7);
    linalg::DenseMatrixOperator      J_next;
    linalg::DenseMatrixOperator      J_hist;
    linalg::DenseLinearSolver        solver;
    state::TimeLinearIntegrator     integrator(problem, J_next, solver);
    femx::problem::TimeLinearization adj_lin;

    Vector<Real> initial(1);
    initial[0] = 0.35;
    integrator.setInitialState(initial);

    TerminalTrackingObjective    obj(problem.dims().nt, 1.1);
    state::TimeReducedFunctional physical(
        integrator, problem, adj_lin, J_next, J_hist, solver, obj);

    Vector<Real> mean{0.4};
    DenseMatrix  perturbations(1, 2);
    perturbations(0, 0) = 0.8;
    perturbations(0, 1) = -0.3;
    const state::EnsembleBasis basis(mean, perturbations);

    const Real                       prior_weight = 0.6;
    state::EnsembleReducedFunctional reduced(
        physical, basis, prior_weight);

    Vector<Real> alpha{0.35, -0.2};
    Vector<Real> prm;
    basis.apply(alpha, prm);

    Vector<Real> physical_grad;
    const Real   physical_value =
        physical.valueGrad(prm, physical_grad);

    Vector<Real> expected_grad;
    basis.applyT(physical_grad, expected_grad);
    for (Index i = 0; i < expected_grad.size(); ++i)
    {
      expected_grad[i] += prior_weight * alpha[i];
    }
    const Real expected_value =
        physical_value
        + 0.5 * prior_weight
              * (alpha[0] * alpha[0] + alpha[1] * alpha[1]);

    Vector<Real> grad;
    const Real   value  = reduced.valueGrad(alpha, grad);
    status             *= near(value, expected_value, 1.0e-14);
    status             *= grad.size() == expected_grad.size();
    for (Index i = 0; i < grad.size(); ++i)
    {
      status *= near(grad[i], expected_grad[i], 1.0e-14);
    }

    const Real eps = 1.0e-6;
    for (Index i = 0; i < alpha.size(); ++i)
    {
      Vector<Real> plus   = alpha;
      Vector<Real> minus  = alpha;
      plus[i]            += eps;
      minus[i]           -= eps;
      const Real fd =
          (reduced.value(plus) - reduced.value(minus)) / (2.0 * eps);
      status *= near(grad[i], fd, 1.0e-7);
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults                results;
  femx::tests::AB2TimeReducedFunctionalTests test;

  results += test.monitorSolveMatchesExpectedHistory();
  results += test.gradientMatchesFiniteDifferenceAndDescends();
  results += test.ensembleReducedFunctionalWrapsTimeReducedFunctional();

  return results.summary();
}
