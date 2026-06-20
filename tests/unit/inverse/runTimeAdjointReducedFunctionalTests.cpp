#include <iostream>

#include <femx/solve/TimeMatrixLinearStateSolver.hpp>
#include <femx/solve/TimeMatrixNewtonStateSolver.hpp>
#include <femx/problem/TimeMatrixResidualEquation.hpp>
#include <femx/problem/TimeResidualEquation.hpp>
#include <femx/solve/DerivativeCheck.hpp>
#include <femx/problem/SumTimeObjectiveFunctional.hpp>
#include <femx/solve/TimeAdjointReducedFunctional.hpp>
#include <femx/problem/TimeObjectiveFunctional.hpp>
#include <femx/solve/TimeReducedFunctional.hpp>
#include <femx/problem/TimeRegularization.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/DenseLinearSolver.hpp>
#include <femx/algebra/backends/native/DenseSystemMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

namespace
{

void resize(Vector<Real>& out,
            Index         size)
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

} // namespace

class ScalarTimeEquation final : public problem::TimeMatrixResidualEquation
{
public:
  Index numSteps() const override
  {
    return 4;
  }

  Index numStates() const override
  {
    return 1;
  }

  Index numParams() const override
  {
    return numSteps();
  }

  Index numRes() const override
  {
    return 1;
  }

  void res(Index               step,
           const Vector<Real>& x_next,
           const Vector<Real>& x,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    resize(out, numRes());
    out[0] = x_next[0] - a() * x[0] - b() * prm[step];
  }

  void assembleNextStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::SystemMatrix& out) const override
  {
    (void) step;
    (void) x_next;
    (void) x;
    (void) prm;
    out.resize(numRes(), numStates());
    out.setZero();
    out.set(0, 0, 1.0);
  }

  void assemblePrevStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::SystemMatrix& out) const override
  {
    (void) step;
    (void) x_next;
    (void) x;
    (void) prm;
    out.resize(numRes(), numStates());
    out.setZero();
    out.set(0, 0, -a());
  }

  void assembleParamJac(Index                 step,
                        const Vector<Real>&   x_next,
                        const Vector<Real>&   x,
                        const Vector<Real>&   prm,
                        algebra::SystemMatrix& out) const override
  {
    (void) x_next;
    (void) x;
    (void) prm;
    out.resize(numRes(), numParams());
    out.setZero();
    out.set(0, step, -b());
  }

  static Real a()
  {
    return 0.75;
  }

  static Real b()
  {
    return 1.30;
  }
};

class ScalarTrackingObjective final
  : public problem::TimeObjectiveFunctional
{
public:
  ScalarTrackingObjective()
    : target_(numSteps() + 1)
  {
    target_[0] = 0.2;
    target_[1] = 0.05;
    target_[2] = -0.25;
    target_[3] = 0.10;
    target_[4] = 0.45;
  }

  Index numSteps() const override
  {
    return 4;
  }

  Index numStates() const override
  {
    return 1;
  }

  Index numParams() const override
  {
    return numSteps();
  }

  Real value(const solve::TimeStateTrajectory& tr,
             const Vector<Real>&            prm) const override
  {
    (void) prm;
    Real value_out = 0.0;
    for (Index level = 1; level <= numSteps(); ++level)
    {
      const Real diff  = tr[level][0] - target_[level];
      value_out       += 0.5 * diff * diff;
    }
    return value_out;
  }

  void stateGrad(Index                          level,
                 const solve::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override
  {
    (void) prm;
    resize(out, numStates());
    if (level > 0)
    {
      out[0] = tr[level][0] - target_[level];
    }
  }

  void paramGrad(const solve::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override
  {
    (void) tr;
    (void) prm;
    resize(out, numParams());
  }

private:
  Vector<Real> target_;
};

class InitialScalarTimeEquation final : public problem::TimeMatrixResidualEquation
{
public:
  Index numSteps() const override
  {
    return 4;
  }

  Index numStates() const override
  {
    return 1;
  }

  Index numParams() const override
  {
    return numSteps() + 1;
  }

  Index numRes() const override
  {
    return 1;
  }

  void res(Index               step,
           const Vector<Real>& x_next,
           const Vector<Real>& x,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    resize(out, numRes());
    out[0] = x_next[0] - a() * x[0] - b() * prm[step + 1];
  }

  void assembleNextStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::SystemMatrix& out) const override
  {
    (void) step;
    (void) x_next;
    (void) x;
    (void) prm;
    out.resize(numRes(), numStates());
    out.setZero();
    out.set(0, 0, 1.0);
  }

  void assemblePrevStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::SystemMatrix& out) const override
  {
    (void) step;
    (void) x_next;
    (void) x;
    (void) prm;
    out.resize(numRes(), numStates());
    out.setZero();
    out.set(0, 0, -a());
  }

  void assembleParamJac(Index                 step,
                        const Vector<Real>&   x_next,
                        const Vector<Real>&   x,
                        const Vector<Real>&   prm,
                        algebra::SystemMatrix& out) const override
  {
    (void) x_next;
    (void) x;
    (void) prm;
    out.resize(numRes(), numParams());
    out.setZero();
    out.set(0, step + 1, -b());
  }

  static Real a()
  {
    return 0.75;
  }

  static Real b()
  {
    return 1.30;
  }
};

class InitialParameterStateSolver final : public solve::TimeStateSolver
{
public:
  explicit InitialParameterStateSolver(
      solve::TimeMatrixLinearStateSolver& solver)
    : solver_(solver)
  {
  }

  Index numSteps() const override
  {
    return solver_.numSteps();
  }

  Index numStates() const override
  {
    return solver_.numStates();
  }

  Index numParams() const override
  {
    return solver_.numParams();
  }

  void solve(const Vector<Real>&      prm,
             solve::TimeStateTrajectory& tr) override
  {
    Vector<Real> init(numStates());
    init[0] = prm[0];
    solver_.setInitialState(init);
    solver_.solve(prm, tr);
  }

private:
  solve::TimeMatrixLinearStateSolver& solver_;
};

class InitialScalarTrackingObjective final
  : public problem::TimeObjectiveFunctional
{
public:
  InitialScalarTrackingObjective()
    : target_(numSteps() + 1)
  {
    target_[0] = 0.15;
    target_[1] = 0.05;
    target_[2] = -0.25;
    target_[3] = 0.10;
    target_[4] = 0.45;
  }

  Index numSteps() const override
  {
    return 4;
  }

  Index numStates() const override
  {
    return 1;
  }

  Index numParams() const override
  {
    return numSteps() + 1;
  }

  Real value(const solve::TimeStateTrajectory& tr,
             const Vector<Real>&            prm) const override
  {
    (void) prm;
    Real value_out = 0.0;
    for (Index level = 0; level <= numSteps(); ++level)
    {
      const Real diff  = tr[level][0] - target_[level];
      value_out       += 0.5 * diff * diff;
    }
    return value_out;
  }

  void stateGrad(Index                          level,
                 const solve::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override
  {
    (void) prm;
    resize(out, numStates());
    out[0] = tr[level][0] - target_[level];
  }

  void paramGrad(const solve::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override
  {
    (void) tr;
    (void) prm;
    resize(out, numParams());
  }

private:
  Vector<Real> target_;
};

class TimeAdjointReducedFunctionalTests : public TestBase
{
public:
  TestOutcome timeMatrixNewtonStateSolverSolvesTrajectory()
  {
    TestStatus status;
    status = true;

    ScalarTimeEquation              eq;
    algebra::DenseLinearSolver       lin_solver;
    algebra::DenseSystemMatrix       next_state_jac;
    solve::TimeMatrixNewtonStateSolver state_solver(
        eq, next_state_jac, lin_solver);

    Vector<Real> init(eq.numStates());
    init[0] = 0.2;
    state_solver.setInitialState(init);

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.10;
    prm[1] = -0.20;
    prm[2] = 0.05;
    prm[3] = 0.30;

    solve::TimeStateTrajectory tr;
    state_solver.solve(prm, tr);

    status *= (tr.numSteps() == eq.numSteps());
    status *= (tr.numStates() == eq.numStates());

    Real expected  = init[0];
    status        *= isEqual(tr[0][0], expected);
    for (Index step = 0; step < eq.numSteps(); ++step)
    {
      expected =
          ScalarTimeEquation::a() * expected
          + ScalarTimeEquation::b() * prm[step];
      status *= isEqual(tr[step + 1][0], expected);

      Vector<Real> res;
      eq.res(step, tr[step + 1], tr[step], prm, res);
      status *= isEqual(res[0], 0.0);
    }

    return status.report(__func__);
  }

  TestOutcome timeMatrixLinearStateSolverSolvesTrajectory()
  {
    TestStatus status;
    status = true;

    ScalarTimeEquation              eq;
    algebra::DenseLinearSolver       lin_solver;
    algebra::DenseSystemMatrix       next_state_jac;
    solve::TimeMatrixLinearStateSolver state_solver(
        eq, next_state_jac, lin_solver);

    Vector<Real> init(eq.numStates());
    init[0] = 0.2;
    state_solver.setInitialState(init);

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.10;
    prm[1] = -0.20;
    prm[2] = 0.05;
    prm[3] = 0.30;

    solve::TimeStateTrajectory tr;
    state_solver.solve(prm, tr);

    status *= (tr.numSteps() == eq.numSteps());
    status *= (tr.numStates() == eq.numStates());

    Real expected  = init[0];
    status        *= isEqual(tr[0][0], expected);
    for (Index step = 0; step < eq.numSteps(); ++step)
    {
      expected =
          ScalarTimeEquation::a() * expected
          + ScalarTimeEquation::b() * prm[step];
      status *= isEqual(tr[step + 1][0], expected);

      Vector<Real> res;
      eq.res(step, tr[step + 1], tr[step], prm, res);
      status *= isEqual(res[0], 0.0);
    }

    return status.report(__func__);
  }

  TestOutcome computesValueAndAdjointGradient()
  {
    TestStatus status;
    status = true;

    ScalarTimeEquation              eq;
    algebra::DenseLinearSolver       lin_solver;
    algebra::DenseSystemMatrix       next_state_jac;
    solve::TimeMatrixLinearStateSolver state_solver(
        eq, next_state_jac, lin_solver);
    ScalarTrackingObjective     tracking;
    problem::TimeRegularization reg(
        eq.numSteps(), eq.numStates(), eq.numSteps(), 1, 0.2, 0.05);

    Vector<Real> init(eq.numStates());
    init[0] = 0.2;
    state_solver.setInitialState(init);

    problem::SumTimeObjectiveFunctional obj(
        eq.numSteps(), eq.numStates(), eq.numParams());
    obj.add(tracking).add(reg);

    solve::TimeAdjointReducedFunctional functional(
        state_solver, eq, lin_solver, obj);

    status *= (functional.numParams() == eq.numParams());

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.10;
    prm[1] = -0.20;
    prm[2] = 0.05;
    prm[3] = 0.30;

    solve::TimeStateTrajectory tr;
    state_solver.solve(prm, tr);
    status *= isEqual(functional.value(prm),
                      obj.value(tr, prm));

    Vector<Real> grad;
    const Real   value_from_value_grad  = functional.valueGrad(prm, grad);
    status                             *= isEqual(value_from_value_grad,
                      functional.value(prm));
    status                             *= (grad.size() == functional.numParams());

    Vector<Real> dir(eq.numParams());
    dir[0] = 0.30;
    dir[1] = -0.50;
    dir[2] = 0.20;
    dir[3] = -0.40;

    const solve::DerivativeCheck check(1.0e-6);
    status *= check.reducedGrad(functional, prm, dir)
                  .passed(1.0e-7, 1.0e-7);

    return status.report(__func__);
  }

  TestOutcome assembledFunctionalComputesValueAndAdjointGradient()
  {
    TestStatus status;
    status = true;

    ScalarTimeEquation              eq;
    algebra::DenseLinearSolver       lin_solver;
    algebra::DenseSystemMatrix       state_next_jac;
    solve::TimeMatrixLinearStateSolver state_solver(
        eq, state_next_jac, lin_solver);
    ScalarTrackingObjective     tracking;
    problem::TimeRegularization reg(
        eq.numSteps(), eq.numStates(), eq.numSteps(), 1, 0.2, 0.05);

    Vector<Real> init(eq.numStates());
    init[0] = 0.2;
    state_solver.setInitialState(init);

    problem::SumTimeObjectiveFunctional obj(
        eq.numSteps(), eq.numStates(), eq.numParams());
    obj.add(tracking).add(reg);

    algebra::DenseSystemMatrix      adj_next_jac;
    algebra::DenseSystemMatrix      adj_prev_jac;
    solve::TimeReducedFunctional functional(
        state_solver, eq, adj_next_jac, adj_prev_jac, lin_solver, obj);

    status *= (functional.numParams() == eq.numParams());

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.10;
    prm[1] = -0.20;
    prm[2] = 0.05;
    prm[3] = 0.30;

    solve::TimeStateTrajectory tr;
    state_solver.solve(prm, tr);
    status *= isEqual(functional.value(prm),
                      obj.value(tr, prm));

    Vector<Real> grad;
    const Real   value_from_value_grad  = functional.valueGrad(prm, grad);
    status                             *= isEqual(value_from_value_grad,
                      functional.value(prm));
    status                             *= (grad.size() == functional.numParams());

    Vector<Real> dir(eq.numParams());
    dir[0] = 0.30;
    dir[1] = -0.50;
    dir[2] = 0.20;
    dir[3] = -0.40;

    const solve::DerivativeCheck check(1.0e-6);
    status *= check.reducedGrad(functional, prm, dir)
                  .passed(1.0e-7, 1.0e-7);

    return status.report(__func__);
  }

  TestOutcome assembledFunctionalHandlesInitialStateParameters()
  {
    TestStatus status;
    status = true;

    InitialScalarTimeEquation       eq;
    algebra::DenseLinearSolver       lin_solver;
    algebra::DenseSystemMatrix       state_next_jac;
    solve::TimeMatrixLinearStateSolver inner_state_solver(
        eq, state_next_jac, lin_solver);
    InitialParameterStateSolver    state_solver(inner_state_solver);
    InitialScalarTrackingObjective tracking;

    problem::SumTimeObjectiveFunctional obj(
        eq.numSteps(), eq.numStates(), eq.numParams());
    obj.add(tracking);

    algebra::DenseSystemMatrix      adj_next_jac;
    algebra::DenseSystemMatrix      adj_prev_jac;
    solve::TimeReducedFunctional functional(
        state_solver, eq, adj_next_jac, adj_prev_jac, lin_solver, obj);
    functional.setInitialStateParamJacT(
        [](const Vector<Real>& prm,
           const Vector<Real>& state_grad,
           Vector<Real>&       out)
        {
          resize(out, prm.size());
          out[0] = state_grad[0];
        });

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.20;
    prm[1] = 0.10;
    prm[2] = -0.20;
    prm[3] = 0.05;
    prm[4] = 0.30;

    solve::TimeStateTrajectory tr;
    state_solver.solve(prm, tr);
    status *= isEqual(tr[0][0], prm[0]);
    status *= isEqual(functional.value(prm),
                      obj.value(tr, prm));

    Vector<Real> grad;
    const Real   value_from_value_grad  = functional.valueGrad(prm, grad);
    status                             *= isEqual(value_from_value_grad, functional.value(prm));
    status                             *= (grad.size() == functional.numParams());

    Vector<Real> dir(eq.numParams());
    dir[0] = -0.40;
    dir[1] = 0.30;
    dir[2] = -0.50;
    dir[3] = 0.20;
    dir[4] = -0.10;

    const solve::DerivativeCheck check(1.0e-6);
    status *= check.reducedGrad(functional, prm, dir)
                  .passed(1.0e-7, 1.0e-7);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running time adjoint reduced functional tests:\n";

  femx::tests::TimeAdjointReducedFunctionalTests test;

  femx::tests::TestingResults result;
  result += test.timeMatrixNewtonStateSolverSolvesTrajectory();
  result += test.timeMatrixLinearStateSolverSolvesTrajectory();
  result += test.computesValueAndAdjointGradient();
  result += test.assembledFunctionalComputesValueAndAdjointGradient();
  result += test.assembledFunctionalHandlesInitialStateParameters();

  return result.summary();
}
