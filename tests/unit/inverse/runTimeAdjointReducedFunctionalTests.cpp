#include <iostream>

#include <femx/eq/TimeMatrixLinearStateSolver.hpp>
#include <femx/eq/TimeMatrixNewtonStateSolver.hpp>
#include <femx/eq/TimeMatrixResidualEquation.hpp>
#include <femx/eq/TimeResidualEquation.hpp>
#include <femx/inverse/DerivativeCheck.hpp>
#include <femx/inverse/SumTimeObjectiveFunctional.hpp>
#include <femx/inverse/TimeAdjointReducedFunctional.hpp>
#include <femx/inverse/TimeObjectiveFunctional.hpp>
#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/inverse/TimeRegularization.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/DenseLinearSolver.hpp>
#include <femx/system/native/DenseSystemMatrix.hpp>
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

class ScalarTimeEquation final : public eq::TimeMatrixResidualEquation
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
                            system::SystemMatrix& out) const override
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
                                system::SystemMatrix& out) const override
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
                        system::SystemMatrix& out) const override
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
  : public inverse::TimeObjectiveFunctional
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

  Real value(const eq::TimeStateTrajectory& tr,
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
                 const eq::TimeStateTrajectory& tr,
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

  void paramGrad(const eq::TimeStateTrajectory& tr,
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

class InitialScalarTimeEquation final : public eq::TimeMatrixResidualEquation
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
                            system::SystemMatrix& out) const override
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
                            system::SystemMatrix& out) const override
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
                        system::SystemMatrix& out) const override
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

class InitialParameterStateSolver final : public eq::TimeStateSolver
{
public:
  explicit InitialParameterStateSolver(
      eq::TimeMatrixLinearStateSolver& solver)
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

  void solve(const Vector<Real>&  prm,
             eq::TimeStateTrajectory& tr) override
  {
    Vector<Real> init(numStates());
    init[0] = prm[0];
    solver_.setInitialState(init);
    solver_.solve(prm, tr);
  }

private:
  eq::TimeMatrixLinearStateSolver& solver_;
};

class InitialScalarTrackingObjective final
  : public inverse::TimeObjectiveFunctional
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

  Real value(const eq::TimeStateTrajectory& tr,
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
                 const eq::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override
  {
    (void) prm;
    resize(out, numStates());
    out[0] = tr[level][0] - target_[level];
  }

  void paramGrad(const eq::TimeStateTrajectory& tr,
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
    system::DenseLinearSolver       lin_solver;
    system::DenseSystemMatrix       next_state_jac;
    eq::TimeMatrixNewtonStateSolver state_solver(
        eq, next_state_jac, lin_solver);

    Vector<Real> init(eq.numStates());
    init[0] = 0.2;
    state_solver.setInitialState(init);

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.10;
    prm[1] = -0.20;
    prm[2] = 0.05;
    prm[3] = 0.30;

    eq::TimeStateTrajectory tr;
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
    system::DenseLinearSolver       lin_solver;
    system::DenseSystemMatrix       next_state_jac;
    eq::TimeMatrixLinearStateSolver state_solver(
        eq, next_state_jac, lin_solver);

    Vector<Real> init(eq.numStates());
    init[0] = 0.2;
    state_solver.setInitialState(init);

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.10;
    prm[1] = -0.20;
    prm[2] = 0.05;
    prm[3] = 0.30;

    eq::TimeStateTrajectory tr;
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
    system::DenseLinearSolver       lin_solver;
    system::DenseSystemMatrix       next_state_jac;
    eq::TimeMatrixLinearStateSolver state_solver(
        eq, next_state_jac, lin_solver);
    ScalarTrackingObjective              tracking;
    inverse::TimeRegularization reg(
        eq.numSteps(), eq.numStates(), eq.numSteps(), 1, 0.2, 0.05);

    Vector<Real> init(eq.numStates());
    init[0] = 0.2;
    state_solver.setInitialState(init);

    inverse::SumTimeObjectiveFunctional obj(
        eq.numSteps(), eq.numStates(), eq.numParams());
    obj.add(tracking).add(reg);

    inverse::TimeAdjointReducedFunctional functional(
        state_solver, eq, lin_solver, obj);

    status *= (functional.numParams() == eq.numParams());

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.10;
    prm[1] = -0.20;
    prm[2] = 0.05;
    prm[3] = 0.30;

    eq::TimeStateTrajectory tr;
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

    const inverse::DerivativeCheck check(1.0e-6);
    status *= check.reducedGrad(functional, prm, dir)
                  .passed(1.0e-7, 1.0e-7);

    return status.report(__func__);
  }

  TestOutcome assembledFunctionalComputesValueAndAdjointGradient()
  {
    TestStatus status;
    status = true;

    ScalarTimeEquation              eq;
    system::DenseLinearSolver       lin_solver;
    system::DenseSystemMatrix       state_next_jac;
    eq::TimeMatrixLinearStateSolver state_solver(
        eq, state_next_jac, lin_solver);
    ScalarTrackingObjective tracking;
    inverse::TimeRegularization reg(
        eq.numSteps(), eq.numStates(), eq.numSteps(), 1, 0.2, 0.05);

    Vector<Real> init(eq.numStates());
    init[0] = 0.2;
    state_solver.setInitialState(init);

    inverse::SumTimeObjectiveFunctional obj(
        eq.numSteps(), eq.numStates(), eq.numParams());
    obj.add(tracking).add(reg);

    system::DenseSystemMatrix adj_next_jac;
    system::DenseSystemMatrix adj_prev_jac;
    inverse::TimeReducedFunctional functional(
        state_solver, eq, adj_next_jac, adj_prev_jac, lin_solver, obj);

    status *= (functional.numParams() == eq.numParams());

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.10;
    prm[1] = -0.20;
    prm[2] = 0.05;
    prm[3] = 0.30;

    eq::TimeStateTrajectory tr;
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

    const inverse::DerivativeCheck check(1.0e-6);
    status *= check.reducedGrad(functional, prm, dir)
                  .passed(1.0e-7, 1.0e-7);

    return status.report(__func__);
  }

  TestOutcome assembledFunctionalHandlesInitialStateParameters()
  {
    TestStatus status;
    status = true;

    InitialScalarTimeEquation      eq;
    system::DenseLinearSolver      lin_solver;
    system::DenseSystemMatrix      state_next_jac;
    eq::TimeMatrixLinearStateSolver inner_state_solver(
        eq, state_next_jac, lin_solver);
    InitialParameterStateSolver state_solver(inner_state_solver);
    InitialScalarTrackingObjective tracking;

    inverse::SumTimeObjectiveFunctional obj(
        eq.numSteps(), eq.numStates(), eq.numParams());
    obj.add(tracking);

    system::DenseSystemMatrix adj_next_jac;
    system::DenseSystemMatrix adj_prev_jac;
    inverse::TimeReducedFunctional functional(
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

    eq::TimeStateTrajectory tr;
    state_solver.solve(prm, tr);
    status *= isEqual(tr[0][0], prm[0]);
    status *= isEqual(functional.value(prm),
                      obj.value(tr, prm));

    Vector<Real> grad;
    const Real   value_from_value_grad = functional.valueGrad(prm, grad);
    status *= isEqual(value_from_value_grad, functional.value(prm));
    status *= (grad.size() == functional.numParams());

    Vector<Real> dir(eq.numParams());
    dir[0] = -0.40;
    dir[1] = 0.30;
    dir[2] = -0.50;
    dir[3] = 0.20;
    dir[4] = -0.10;

    const inverse::DerivativeCheck check(1.0e-6);
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
