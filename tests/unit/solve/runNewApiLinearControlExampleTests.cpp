#include <cmath>
#include <iostream>

#include <examples/inverse/linear-control-new-api/Problem.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class NewApiLinearControlExampleTests : public TestBase
{
public:
  TestOutcome reducedFunctionalGradientMatchesFiniteDifference()
  {
    TestStatus status;
    status = true;

    examples_inverse_linear_control_new_api::LinearControlSetup setup;

    Vector<Real> prm =
        examples_inverse_linear_control_new_api::makeInitialParams();
    Vector<Real> grad;
    const Real   value = setup.functional.valueGrad(prm, grad);

    status *= (grad.size() == 2);
    status *= (value > 0.0);

    constexpr Real eps = 1.0e-6;
    for (Index i = 0; i < prm.size(); ++i)
    {
      Vector<Real> plus  = prm;
      Vector<Real> minus = prm;
      plus[i] += eps;
      minus[i] -= eps;

      const Real fd =
          (setup.functional.value(plus) - setup.functional.value(minus))
          / (2.0 * eps);
      status *= near(grad[i], fd, 1.0e-7);
    }

    return status.report(__func__);
  }

  TestOutcome exampleObjectsUseNewProblemAndSolveApi()
  {
    TestStatus status;
    status = true;

    examples_inverse_linear_control_new_api::LinearControlSetup setup;
    status *= (setup.state_solver.numStates() == 2);
    status *= (setup.state_solver.numParams() == 2);
    status *= (setup.functional.numParams() == 2);

    Vector<Real> prm =
        examples_inverse_linear_control_new_api::makeInitialParams();
    Vector<Real> state;
    setup.state_solver.solve(prm, state);

    Vector<Real> residual;
    setup.problem.residual(state, prm, residual);
    status *= near(residual[0], 0.0, 1.0e-10);
    status *= near(residual[1], 0.0, 1.0e-10);

    return status.report(__func__);
  }

private:
  static bool near(Real a, Real b, Real tolerance)
  {
    return std::abs(a - b) <= tolerance * (1.0 + std::abs(b));
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running new API linear-control example tests:\n";

  femx::tests::NewApiLinearControlExampleTests test;

  femx::tests::TestingResults result;
  result += test.reducedFunctionalGradientMatchesFiniteDifference();
  result += test.exampleObjectsUseNewProblemAndSolveApi();

  return result.summary();
}
