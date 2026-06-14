#include <iostream>

#include <femx/inverse/AdjointReducedFunctional.hpp>
#include <femx/inverse/DerivativeCheck.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearResidualEquation final : public eq::ResidualEquation
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

  void res(const Vector& state,
           const Vector& params,
           Vector&       out) const override
  {
    resize(out, numRes());
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * params[0] - 2.0 * params[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * params[0] + 4.0 * params[1];
  }

  void applyStateJac(const Vector& state,
                     const Vector& params,
                     const Vector& dir,
                     Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numRes());
    out[0] = 2.0 * dir[0] + 3.0 * dir[1];
    out[1] = 7.0 * dir[0] + 11.0 * dir[1];
  }

  void applyStateJacT(const Vector& state,
                      const Vector& params,
                      const Vector& lambda,
                      Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numStates());
    out[0] = 2.0 * lambda[0] + 7.0 * lambda[1];
    out[1] = 3.0 * lambda[0] + 11.0 * lambda[1];
  }

  void applyParamJac(const Vector& state,
                     const Vector& params,
                     const Vector& dir,
                     Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numRes());
    out[0] = 5.0 * dir[0] - 2.0 * dir[1];
    out[1] = 13.0 * dir[0] + 4.0 * dir[1];
  }

  void applyParamJacT(const Vector& state,
                      const Vector& params,
                      const Vector& lambda,
                      Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numParams());
    out[0] = 5.0 * lambda[0] + 13.0 * lambda[1];
    out[1] = -2.0 * lambda[0] + 4.0 * lambda[1];
  }

private:
  static void resize(Vector& out, Index size)
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

class LinearStateSolver final : public eq::StateSolver
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

  void solve(const Vector& params, Vector& state) override
  {
    resize(state, numStates());

    const Real b0 = 5.0 * params[0] - 2.0 * params[1];
    const Real b1 = 13.0 * params[0] + 4.0 * params[1];

    state[0] = -(11.0 * b0 - 3.0 * b1);
    state[1] = -(-7.0 * b0 + 2.0 * b1);
  }

private:
  static void resize(Vector& out, Index size)
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

class LinearAdjointSolver final : public inverse::AdjointSolver
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

  void solve(const Vector& state,
             const Vector& params,
             const Vector& rhs,
             Vector&       adjoint) override
  {
    (void) state;
    (void) params;
    resize(adjoint, numRes());

    const Real det = 2.0 * 11.0 - 7.0 * 3.0;
    adjoint[0]     = (11.0 * rhs[0] - 7.0 * rhs[1]) / det;
    adjoint[1]     = (-3.0 * rhs[0] + 2.0 * rhs[1]) / det;
  }

private:
  static void resize(Vector& out, Index size)
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

class TrackingObjective final : public inverse::ObjectiveFunctional
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

  Real value(const Vector& state,
             const Vector& params) const override
  {
    const Real e0 = state[0] - target_[0];
    const Real e1 = state[1] - target_[1];
    return 0.5 * (e0 * e0 + e1 * e1)
           + 0.5 * regularization_ * (params[0] * params[0] + params[1] * params[1]);
  }

  void stateGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    (void) params;
    resize(out, numStates());
    out[0] = state[0] - target_[0];
    out[1] = state[1] - target_[1];
  }

  void paramGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    (void) state;
    resize(out, numParams());
    out[0] = regularization_ * params[0];
    out[1] = regularization_ * params[1];
  }

private:
  static void resize(Vector& out, Index size)
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

class AdjointReducedFunctionalTests : public TestBase
{
public:
  TestOutcome computesReducedValueAndGradient()
  {
    TestStatus status;
    status = true;

    LinearStateSolver      state_solver;
    LinearAdjointSolver    adj_solver;
    LinearResidualEquation equation;
    TrackingObjective      objective;

    inverse::AdjointReducedFunctional functional(
        state_solver, adj_solver, equation, objective);

    status *= (functional.numParams() == 2);

    Vector params(2);
    params[0] = 0.05;
    params[1] = -0.02;

    Vector state;
    state_solver.solve(params, state);

    status *= isEqual(functional.value(params),
                      objective.value(state, params));

    Vector     grad;
    const Real value_from_value_grad  = functional.valueGrad(params, grad);
    status                           *= isEqual(value_from_value_grad, functional.value(params));
    status                           *= (grad.size() == functional.numParams());

    Vector dir(2);
    dir[0] = -0.7;
    dir[1] = 0.4;

    const inverse::DerivativeCheck check(1.0e-6);
    status *= check.reducedGrad(functional, params, dir)
                  .passed(1.0e-7, 1.0e-7);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running adjoint reduced functional tests:\n";

  femx::tests::AdjointReducedFunctionalTests test;

  femx::tests::TestingResults result;
  result += test.computesReducedValueAndGradient();

  return result.summary();
}
