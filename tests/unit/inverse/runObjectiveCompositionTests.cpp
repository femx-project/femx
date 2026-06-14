#include <iostream>
#include <stdexcept>

#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/inverse/QuadraticParameterRegularization.hpp>
#include <femx/inverse/SumObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearObjective final : public inverse::ObjectiveFunctional
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
    return 2.0 * state[0] - state[1] + 3.0 * params[0] - 4.0 * params[1];
  }

  void stateGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numStates());
    out[0] = 2.0;
    out[1] = -1.0;
  }

  void paramGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numParams());
    out[0] = 3.0;
    out[1] = -4.0;
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

class ObjectiveCompositionTests : public TestBase
{
public:
  TestOutcome regularizationValueAndGradient()
  {
    TestStatus status;
    status = true;

    Vector reference(2);
    reference[0] = 1.0;
    reference[1] = -2.0;

    const inverse::QuadraticParameterRegularization regularization(
        2, reference, 0.5);

    Vector state(2);
    state[0] = -1.0;
    state[1] = 3.0;

    Vector params(2);
    params[0] = 3.0;
    params[1] = -5.0;

    status *= isEqual(regularization.value(state, params), 3.25);

    Vector grad;
    regularization.stateGrad(state, params, grad);
    status *= isEqual(grad[0], 0.0);
    status *= isEqual(grad[1], 0.0);

    regularization.paramGrad(state, params, grad);
    status *= isEqual(grad[0], 1.0);
    status *= isEqual(grad[1], -1.5);

    return status.report(__func__);
  }

  TestOutcome sumObjectiveAddsTerms()
  {
    TestStatus status;
    status = true;

    LinearObjective linear;

    Vector reference(2);
    reference[0] = 1.0;
    reference[1] = -2.0;

    const inverse::QuadraticParameterRegularization regularization(
        2, reference, 0.5);

    inverse::SumObjectiveFunctional sum(2, 2);
    sum.add(linear).add(regularization);

    Vector state(2);
    state[0] = -1.0;
    state[1] = 3.0;

    Vector params(2);
    params[0] = 3.0;
    params[1] = -5.0;

    status *= isEqual(sum.value(state, params),
                      linear.value(state, params)
                          + regularization.value(state, params));

    Vector grad;
    sum.stateGrad(state, params, grad);
    status *= isEqual(grad[0], 2.0);
    status *= isEqual(grad[1], -1.0);

    sum.paramGrad(state, params, grad);
    status *= isEqual(grad[0], 4.0);
    status *= isEqual(grad[1], -5.5);

    return status.report(__func__);
  }

  TestOutcome sumRejectsMismatchedTerm()
  {
    TestStatus status;
    status = true;

    Vector reference(1);
    reference[0] = 0.0;
    const inverse::QuadraticParameterRegularization regularization(
        2, reference, 1.0);

    inverse::SumObjectiveFunctional sum(2, 2);

    bool threw = false;
    try
    {
      sum.add(regularization);
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
