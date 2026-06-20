#include <iostream>

#include <femx/problem/ResidualEquation.hpp>
#include <femx/solve/DerivativeCheck.hpp>
#include <femx/problem/ObjectiveFunctional.hpp>
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

class QuadraticObjective final : public problem::ObjectiveFunctional
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
    return 0.5 * state[0] * state[0]
           + 2.0 * state[1] * state[1]
           + 3.0 * state[0] * prm[0]
           - 2.0 * state[1] * prm[1]
           + 2.5 * prm[0] * prm[0]
           + 3.5 * prm[1] * prm[1];
  }

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    resize(out, numStates());
    out[0] = state[0] + 3.0 * prm[0];
    out[1] = 4.0 * state[1] - 2.0 * prm[1];
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    resize(out, numParams());
    out[0] = 3.0 * state[0] + 5.0 * prm[0];
    out[1] = -2.0 * state[1] + 7.0 * prm[1];
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

class DerivativeCheckTests : public TestBase
{
public:
  TestOutcome checksObjectiveAndResidualDerivatives()
  {
    TestStatus status;
    status = true;

    const solve::DerivativeCheck check(1.0e-6);

    Vector<Real> state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector<Real> obj_prm(2);
    obj_prm[0] = 2.0;
    obj_prm[1] = -1.5;

    Vector<Real> state_dir(2);
    state_dir[0] = 1.5;
    state_dir[1] = -2.0;

    Vector<Real> param_dir(2);
    param_dir[0] = -0.75;
    param_dir[1] = 0.5;

    QuadraticObjective obj;
    status *= check.objStateGrad(obj,
                                 state,
                                 obj_prm,
                                 state_dir)
                  .passed(1.0e-8, 1.0e-8);
    status *= check.objParamGrad(obj,
                                 state,
                                 obj_prm,
                                 param_dir)
                  .passed(1.0e-8, 1.0e-8);

    Vector<Real> eq_prm(1);
    eq_prm[0] = 2.0;

    Vector<Real> eq_param_dir(1);
    eq_param_dir[0] = -0.75;

    Vector<Real> lambda(2);
    lambda[0] = -3.0;
    lambda[1] = 4.0;

    LinearResidualEquation eq;
    status *= check.resStateJac(eq,
                                state,
                                eq_prm,
                                state_dir)
                  .passed(1.0e-8, 1.0e-8);
    status *= check.resParamJac(eq,
                                state,
                                eq_prm,
                                eq_param_dir)
                  .passed(1.0e-8, 1.0e-8);
    status *= check.stateJacT(eq,
                              state,
                              eq_prm,
                              state_dir,
                              lambda)
                  .passed(1.0e-12, 1.0e-12);
    status *= check.paramJacT(eq,
                              state,
                              eq_prm,
                              eq_param_dir,
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
