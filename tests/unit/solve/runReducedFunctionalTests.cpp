#include <iostream>

#include <femx/problem/ProblemResidualAdapter.hpp>
#include <femx/problem/Objective.hpp>
#include <femx/solve/Newton.hpp>
#include <femx/solve/ReducedFunctional.hpp>
#include <femx/algebra/DenseLinearSolver.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearResidualEquation final : public problem::ResidualEquation
{
public:
  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 2;
  }

  Index numRes() const override
  {
    return 2;
  }

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    resize(out, numRes());
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * prm[0] - 2.0 * prm[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * prm[0] + 4.0 * prm[1];
  }

  void applyStateJac(const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numRes());
    out[0] = 2.0 * dir[0] + 3.0 * dir[1];
    out[1] = 7.0 * dir[0] + 11.0 * dir[1];
  }

  void applyStateJacT(const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numStates());
    out[0] = 2.0 * lambda[0] + 7.0 * lambda[1];
    out[1] = 3.0 * lambda[0] + 11.0 * lambda[1];
  }

  void applyParamJac(const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numRes());
    out[0] = 5.0 * dir[0] - 2.0 * dir[1];
    out[1] = 13.0 * dir[0] + 4.0 * dir[1];
  }

  void applyParamJacT(const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numParams());
    out[0] = 5.0 * lambda[0] + 13.0 * lambda[1];
    out[1] = -2.0 * lambda[0] + 4.0 * lambda[1];
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

class TrackingObjective final : public problem::Objective
{
public:
  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 2;
  }

  Real value(const Vector<Real>& state,
             const Vector<Real>& prm) const override
  {
    const Real e0 = state[0] - target_[0];
    const Real e1 = state[1] - target_[1];
    return 0.5 * (e0 * e0 + e1 * e1)
           + 0.5 * regularization_ * (prm[0] * prm[0] + prm[1] * prm[1]);
  }

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    (void) prm;
    resize(out, numStates());
    out[0] = state[0] - target_[0];
    out[1] = state[1] - target_[1];
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    (void) state;
    resize(out, numParams());
    out[0] = regularization_ * prm[0];
    out[1] = regularization_ * prm[1];
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

private:
  Real target_[2] = {0.25, -0.75};
  Real regularization_{0.25};
};

class ReducedFunctionalTests : public TestBase
{
public:
  TestOutcome computesReducedValueAndGradient()
  {
    TestStatus status;
    status = true;

    LinearResidualEquation          eq;
    problem::ResidualEquationProblemAdapter problem(eq);
    problem::ResidualEquationLinearization linearization;
    algebra::DenseLinearSolver         linear_solver;
    solve::Newton                     state_solver(
        problem, linearization, linear_solver);
    TrackingObjective obj;
    solve::ReducedFunctional functional(
        problem, obj, state_solver, linear_solver);

    status *= (functional.numParams() == 2);

    Vector<Real> prm(2);
    prm[0] = 0.05;
    prm[1] = -0.02;

    Vector<Real> state;
    state_solver.solve(prm, state);

    status *= isEqual(functional.value(prm), obj.value(state, prm));

    Vector<Real> grad;
    const Real   value_from_value_grad = functional.valueGrad(prm, grad);
    status *= isEqual(value_from_value_grad, functional.value(prm));
    status *= (grad.size() == functional.numParams());

    Vector<Real> dir(2);
    dir[0] = -0.7;
    dir[1] = 0.4;

    const Real directional = grad[0] * dir[0] + grad[1] * dir[1];
    const Real fd          = finiteDifference(functional, prm, dir);
    status *= (absValue(directional - fd) <= 1.0e-6);

    return status.report(__func__);
  }

private:
  static Real finiteDifference(solve::ReducedFunctional& functional,
                               const Vector<Real>&       prm,
                               const Vector<Real>&       dir)
  {
    const Real eps = 1.0e-6;

    Vector<Real> plus(prm.size());
    Vector<Real> minus(prm.size());
    for (Index i = 0; i < prm.size(); ++i)
    {
      plus[i]  = prm[i] + eps * dir[i];
      minus[i] = prm[i] - eps * dir[i];
    }

    return (functional.value(plus) - functional.value(minus)) / (2.0 * eps);
  }

  static Real absValue(Real value)
  {
    return value < 0.0 ? -value : value;
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running solve reduced functional tests:\n";

  femx::tests::ReducedFunctionalTests test;

  femx::tests::TestingResults result;
  result += test.computesReducedValueAndGradient();

  return result.summary();
}
