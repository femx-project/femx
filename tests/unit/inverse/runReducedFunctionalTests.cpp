#include <iostream>

#include <femx/solve/DerivativeCheck.hpp>
#include <femx/solve/ReducedObjective.hpp>
#include <femx/algebra/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class QuadraticReducedFunctional final : public solve::ReducedObjective
{
public:
  Index numParams() const override
  {
    return 2;
  }

  Real value(const Vector<Real>& prm) override
  {
    return 0.5 * prm[0] * prm[0]
           + 2.0 * prm[0] * prm[1]
           + 3.0 * prm[1] * prm[1]
           - prm[0]
           + 4.0 * prm[1];
  }

  void grad(const Vector<Real>& prm, Vector<Real>& out) override
  {
    resize(out, numParams());
    out[0] = prm[0] + 2.0 * prm[1] - 1.0;
    out[1] = 2.0 * prm[0] + 6.0 * prm[1] + 4.0;
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

class ReducedFunctionalTests : public TestBase
{
public:
  TestOutcome quadraticReducedFunctionalOperations()
  {
    TestStatus status;
    status = true;

    QuadraticReducedFunctional functional;
    status *= (functional.numParams() == 2);

    Vector<Real> prm(2);
    prm[0] = 0.25;
    prm[1] = -0.5;

    const Real expected_value =
        0.5 * prm[0] * prm[0]
        + 2.0 * prm[0] * prm[1]
        + 3.0 * prm[1] * prm[1]
        - prm[0]
        + 4.0 * prm[1];
    status *= isEqual(functional.value(prm), expected_value);

    Vector<Real> grad;
    const Real   value_from_value_grad  = functional.valueGrad(prm, grad);
    status                             *= isEqual(value_from_value_grad, expected_value);
    status                             *= isEqual(grad[0], prm[0] + 2.0 * prm[1] - 1.0);
    status                             *= isEqual(grad[1], 2.0 * prm[0] + 6.0 * prm[1] + 4.0);

    Vector<Real> dir(2);
    dir[0] = -0.75;
    dir[1] = 0.5;

    const solve::DerivativeCheck check(1.0e-6);
    status *= check.reducedGrad(functional, prm, dir)
                  .passed(1.0e-8, 1.0e-8);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running reduced functional tests:\n";

  femx::tests::ReducedFunctionalTests test;

  femx::tests::TestingResults result;
  result += test.quadraticReducedFunctionalOperations();

  return result.summary();
}
