#include <iostream>
#include <stdexcept>

#include <femx/problem/ObjectiveFunctional.hpp>
#include <femx/problem/QuadraticParameterRegularization.hpp>
#include <femx/problem/SumObjectiveFunctional.hpp>
#include <femx/algebra/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearObjective final : public problem::ObjectiveFunctional
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
    return 2.0 * state[0] - state[1] + 3.0 * prm[0] - 4.0 * prm[1];
  }

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numStates());
    out[0] = 2.0;
    out[1] = -1.0;
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numParams());
    out[0] = 3.0;
    out[1] = -4.0;
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

class ObjectiveCompositionTests : public TestBase
{
public:
  TestOutcome regularizationValueAndGradient()
  {
    TestStatus status;
    status = true;

    Vector<Real> reference(2);
    reference[0] = 1.0;
    reference[1] = -2.0;

    const problem::QuadraticParameterRegularization reg(
        2, reference, 0.5);

    Vector<Real> state(2);
    state[0] = -1.0;
    state[1] = 3.0;

    Vector<Real> prm(2);
    prm[0] = 3.0;
    prm[1] = -5.0;

    status *= isEqual(reg.value(state, prm), 3.25);

    Vector<Real> grad;
    reg.stateGrad(state, prm, grad);
    status *= isEqual(grad[0], 0.0);
    status *= isEqual(grad[1], 0.0);

    reg.paramGrad(state, prm, grad);
    status *= isEqual(grad[0], 1.0);
    status *= isEqual(grad[1], -1.5);

    return status.report(__func__);
  }

  TestOutcome sumObjectiveAddsTerms()
  {
    TestStatus status;
    status = true;

    LinearObjective linear;

    Vector<Real> reference(2);
    reference[0] = 1.0;
    reference[1] = -2.0;

    const problem::QuadraticParameterRegularization reg(
        2, reference, 0.5);

    problem::SumObjectiveFunctional sum(2, 2);
    sum.add(linear).add(reg);

    Vector<Real> state(2);
    state[0] = -1.0;
    state[1] = 3.0;

    Vector<Real> prm(2);
    prm[0] = 3.0;
    prm[1] = -5.0;

    status *= isEqual(sum.value(state, prm),
                      linear.value(state, prm)
                          + reg.value(state, prm));

    Vector<Real> grad;
    sum.stateGrad(state, prm, grad);
    status *= isEqual(grad[0], 2.0);
    status *= isEqual(grad[1], -1.0);

    sum.paramGrad(state, prm, grad);
    status *= isEqual(grad[0], 4.0);
    status *= isEqual(grad[1], -5.5);

    return status.report(__func__);
  }

  TestOutcome sumRejectsMismatchedTerm()
  {
    TestStatus status;
    status = true;

    Vector<Real> reference(1);
    reference[0] = 0.0;
    const problem::QuadraticParameterRegularization reg(
        2, reference, 1.0);

    problem::SumObjectiveFunctional sum(2, 2);

    bool threw = false;
    try
    {
      sum.add(reg);
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }

    status *= threw;
    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running objective composition tests:\n";

  femx::tests::ObjectiveCompositionTests test;

  femx::tests::TestingResults result;
  result += test.regularizationValueAndGradient();
  result += test.sumObjectiveAddsTerms();
  result += test.sumRejectsMismatchedTerm();

  return result.summary();
}
