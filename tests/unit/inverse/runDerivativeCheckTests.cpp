#include <iostream>

#include <femx/eq/ResidualEquation.hpp>
#include <femx/inverse/DerivativeCheck.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
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
    return 1;
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
    out[0] = 2.0 * state[0] + 3.0 * state[1] + 5.0 * params[0];
    out[1] = 7.0 * state[0] + 11.0 * state[1] + 13.0 * params[0];
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
    out[0] = 5.0 * dir[0];
    out[1] = 13.0 * dir[0];
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

class QuadraticObjective final : public inverse::ObjectiveFunctional
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
    return 0.5 * state[0] * state[0]
           + 2.0 * state[1] * state[1]
           + 3.0 * state[0] * params[0]
           - 2.0 * state[1] * params[1]
           + 2.5 * params[0] * params[0]
           + 3.5 * params[1] * params[1];
  }

  void stateGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    resize(out, numStates());
    out[0] = state[0] + 3.0 * params[0];
    out[1] = 4.0 * state[1] - 2.0 * params[1];
  }

  void paramGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    resize(out, numParams());
    out[0] = 3.0 * state[0] + 5.0 * params[0];
    out[1] = -2.0 * state[1] + 7.0 * params[1];
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

class DerivativeCheckTests : public TestBase
{
public:
  TestOutcome checksObjectiveAndResidualDerivatives()
  {
    TestStatus status;
    status = true;

    const inverse::DerivativeCheck check(1.0e-6);

    Vector state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector objective_params(2);
    objective_params[0] = 2.0;
    objective_params[1] = -1.5;

    Vector state_direction(2);
    state_direction[0] = 1.5;
    state_direction[1] = -2.0;

    Vector param_direction(2);
    param_direction[0] = -0.75;
    param_direction[1] = 0.5;

    QuadraticObjective objective;
    status *= check.objStateGrad(objective,
                                 state,
                                 objective_params,
                                 state_direction)
                  .passed(1.0e-8, 1.0e-8);
    status *= check.objParamGrad(objective,
                                 state,
                                 objective_params,
                                 param_direction)
                  .passed(1.0e-8, 1.0e-8);

    Vector eq_params(1);
    eq_params[0] = 2.0;

    Vector eq_param_direction(1);
    eq_param_direction[0] = -0.75;

    Vector lambda(2);
    lambda[0] = -3.0;
    lambda[1] = 4.0;

    LinearResidualEquation equation;
    status *= check.resStateJac(equation,
                                state,
                                eq_params,
                                state_direction)
                  .passed(1.0e-8, 1.0e-8);
    status *= check.resParamJac(equation,
                                state,
                                eq_params,
                                eq_param_direction)
                  .passed(1.0e-8, 1.0e-8);
    status *= check.stateJacT(equation,
                              state,
                              eq_params,
                              state_direction,
                              lambda)
                  .passed(1.0e-12, 1.0e-12);
    status *= check.paramJacT(equation,
                              state,
                              eq_params,
                              eq_param_direction,
                              lambda)
                  .passed(1.0e-12, 1.0e-12);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running derivative check tests:\n";

  femx::tests::DerivativeCheckTests test;

  femx::tests::TestingResults result;
  result += test.checksObjectiveAndResidualDerivatives();

  return result.summary();
}
