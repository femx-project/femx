#include <iostream>

#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

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

class ObjectiveFunctionalTests : public TestBase
{
public:
  TestOutcome quadraticObjectiveOperations()
  {
    TestStatus status;
    status = true;

    QuadraticObjective objective;

    status *= (objective.numStates() == 2);
    status *= (objective.numParams() == 2);

    Vector state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector params(2);
    params[0] = 2.0;
    params[1] = -1.5;

    const Real expected_value =
        0.5 * state[0] * state[0]
        + 2.0 * state[1] * state[1]
        + 3.0 * state[0] * params[0]
        - 2.0 * state[1] * params[1]
        + 2.5 * params[0] * params[0]
        + 3.5 * params[1] * params[1];
    status *= isEqual(objective.value(state, params), expected_value);

    Vector out;
    objective.stateGrad(state, params, out);
    status *= isEqual(out[0], state[0] + 3.0 * params[0]);
    status *= isEqual(out[1], 4.0 * state[1] - 2.0 * params[1]);

    objective.paramGrad(state, params, out);
    status *= isEqual(out[0], 3.0 * state[0] + 5.0 * params[0]);
    status *= isEqual(out[1], -2.0 * state[1] + 7.0 * params[1]);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running objective functional tests:\n";

  femx::tests::ObjectiveFunctionalTests test;

  femx::tests::TestingResults result;
  result += test.quadraticObjectiveOperations();

  return result.summary();
}
