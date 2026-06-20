#include <iostream>

#include <femx/problem/ResidualEquation.hpp>
#include <femx/algebra/Vector.hpp>
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
    return 1;
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
    out[0] = 2.0 * state[0] + 3.0 * state[1] + 5.0 * prm[0];
    out[1] = 7.0 * state[0] + 11.0 * state[1] + 13.0 * prm[0];
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
    out[0] = 5.0 * dir[0];
    out[1] = 13.0 * dir[0];
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

    LinearResidualEquation eq;

    status *= (eq.numStates() == 2);
    status *= (eq.numParams() == 1);
    status *= (eq.numRes() == 2);

    Vector<Real> state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector<Real> prm(1);
    prm[0] = 2.0;

    Vector<Real> out;
    eq.res(state, prm, out);
    status *= isEqual(out[0], 2.0 * state[0] + 3.0 * state[1] + 5.0 * prm[0]);
    status *= isEqual(out[1], 7.0 * state[0] + 11.0 * state[1] + 13.0 * prm[0]);

    Vector<Real> state_dir(2);
    state_dir[0] = 1.5;
    state_dir[1] = -2.0;

    eq.applyStateJac(state, prm, state_dir, out);
    status *= isEqual(out[0], 2.0 * state_dir[0] + 3.0 * state_dir[1]);
    status *= isEqual(out[1], 7.0 * state_dir[0] + 11.0 * state_dir[1]);

    Vector<Real> lambda(2);
    lambda[0] = -3.0;
    lambda[1] = 4.0;

    eq.applyStateJacT(state, prm, lambda, out);
    status *= isEqual(out[0], 2.0 * lambda[0] + 7.0 * lambda[1]);
    status *= isEqual(out[1], 3.0 * lambda[0] + 11.0 * lambda[1]);

    Vector<Real> param_dir(1);
    param_dir[0] = -0.75;

    eq.applyParamJac(state, prm, param_dir, out);
    status *= isEqual(out[0], 5.0 * param_dir[0]);
    status *= isEqual(out[1], 13.0 * param_dir[0]);

    eq.applyParamJacT(state, prm, lambda, out);
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
