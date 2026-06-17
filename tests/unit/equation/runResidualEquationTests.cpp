#include <iostream>

#include <femx/eq/ResidualEquation.hpp>
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

  void res(const Vector<Real>& state,
           const Vector<Real>& params,
           Vector<Real>&       out) const override
  {
    resize(out, numRes());
    out[0] = 2.0 * state[0] + 3.0 * state[1] + 5.0 * params[0];
    out[1] = 7.0 * state[0] + 11.0 * state[1] + 13.0 * params[0];
  }

  void applyStateJac(const Vector<Real>& state,
                     const Vector<Real>& params,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numRes());
    out[0] = 2.0 * dir[0] + 3.0 * dir[1];
    out[1] = 7.0 * dir[0] + 11.0 * dir[1];
  }

  void applyStateJacT(const Vector<Real>& state,
                      const Vector<Real>& params,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numStates());
    out[0] = 2.0 * lambda[0] + 7.0 * lambda[1];
    out[1] = 3.0 * lambda[0] + 11.0 * lambda[1];
  }

  void applyParamJac(const Vector<Real>& state,
                     const Vector<Real>& params,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numRes());
    out[0] = 5.0 * dir[0];
    out[1] = 13.0 * dir[0];
  }

  void applyParamJacT(const Vector<Real>& state,
                      const Vector<Real>& params,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numParams());
    out[0] = 5.0 * lambda[0] + 13.0 * lambda[1];
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

class ResidualEquationTests : public TestBase
{
public:
  TestOutcome linearEquationOperations()
  {
    TestStatus status;
    status = true;

    LinearResidualEquation equation;

    status *= (equation.numStates() == 2);
    status *= (equation.numParams() == 1);
    status *= (equation.numRes() == 2);

    Vector<Real> state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector<Real> params(1);
    params[0] = 2.0;

    Vector<Real> out;
    equation.res(state, params, out);
    status *= isEqual(out[0], 2.0 * state[0] + 3.0 * state[1] + 5.0 * params[0]);
    status *= isEqual(out[1], 7.0 * state[0] + 11.0 * state[1] + 13.0 * params[0]);

    Vector<Real> state_dir(2);
    state_dir[0] = 1.5;
    state_dir[1] = -2.0;

    equation.applyStateJac(state, params, state_dir, out);
    status *= isEqual(out[0], 2.0 * state_dir[0] + 3.0 * state_dir[1]);
    status *= isEqual(out[1], 7.0 * state_dir[0] + 11.0 * state_dir[1]);

    Vector<Real> lambda(2);
    lambda[0] = -3.0;
    lambda[1] = 4.0;

    equation.applyStateJacT(state, params, lambda, out);
    status *= isEqual(out[0], 2.0 * lambda[0] + 7.0 * lambda[1]);
    status *= isEqual(out[1], 3.0 * lambda[0] + 11.0 * lambda[1]);

    Vector<Real> param_dir(1);
    param_dir[0] = -0.75;

    equation.applyParamJac(state, params, param_dir, out);
    status *= isEqual(out[0], 5.0 * param_dir[0]);
    status *= isEqual(out[1], 13.0 * param_dir[0]);

    equation.applyParamJacT(state, params, lambda, out);
    status *= isEqual(out[0], 5.0 * lambda[0] + 13.0 * lambda[1]);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running residual equation tests:\n";

  femx::tests::ResidualEquationTests test;

  femx::tests::TestingResults result;
  result += test.linearEquationOperations();

  return result.summary();
}
