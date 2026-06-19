#include <cmath>
#include <iostream>

#include <femx/ad/Forward.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class ForwardADTests : public TestBase
{
public:
  TestOutcome differentiatesElementaryExpression()
  {
    TestStatus status;
    status = true;

    const ad::Forward x = ad::Forward::variable(2.0, 2, 0);
    const ad::Forward y = ad::Forward::variable(3.0, 2, 1);
    const ad::Forward f = x * x * y + ad::sqrt(y) - 4.0 / x;

    status *= isEqual(f.value(), 4.0 * 3.0 + std::sqrt(3.0) - 2.0);
    status *= isEqual(f.derivative(0), 2.0 * 2.0 * 3.0 + 4.0 / (2.0 * 2.0));
    status *= isEqual(f.derivative(1), 2.0 * 2.0 + 0.5 / std::sqrt(3.0));

    return status.report(__func__);
  }

  TestOutcome handlesConstantsAndAbs()
  {
    TestStatus status;
    status = true;

    const ad::Forward x = ad::Forward::variable(-2.0, 1, 0);
    const ad::Forward f = 3.0 + ad::abs(x) * 5.0;

    status *= isEqual(f.value(), 13.0);
    status *= isEqual(f.derivative(0), -5.0);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running forward AD tests:\n";

  femx::tests::ForwardADTests test;

  femx::tests::TestingResults result;
  result += test.differentiatesElementaryExpression();
  result += test.handlesConstantsAndAbs();

  return result.summary();
}
