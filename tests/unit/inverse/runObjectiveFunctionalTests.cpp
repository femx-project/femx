#include <iostream>

#include <femx/problem/ObjectiveFunctional.hpp>
#include <femx/algebra/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

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

class ObjectiveFunctionalTests : public TestBase
{
public:
  TestOutcome quadraticObjectiveOperations()
  {
    TestStatus status;
    status = true;

    QuadraticObjective obj;

    status *= (obj.numStates() == 2);
    status *= (obj.numParams() == 2);

    Vector<Real> state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector<Real> prm(2);
    prm[0] = 2.0;
    prm[1] = -1.5;

    const Real expected_value =
        0.5 * state[0] * state[0]
        + 2.0 * state[1] * state[1]
        + 3.0 * state[0] * prm[0]
        - 2.0 * state[1] * prm[1]
        + 2.5 * prm[0] * prm[0]
        + 3.5 * prm[1] * prm[1];
    status *= isEqual(obj.value(state, prm), expected_value);

    Vector<Real> out;
    obj.stateGrad(state, prm, out);
    status *= isEqual(out[0], state[0] + 3.0 * prm[0]);
    status *= isEqual(out[1], 4.0 * state[1] - 2.0 * prm[1]);

    obj.paramGrad(state, prm, out);
    status *= isEqual(out[0], 3.0 * state[0] + 5.0 * prm[0]);
    status *= isEqual(out[1], -2.0 * state[1] + 7.0 * prm[1]);

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
