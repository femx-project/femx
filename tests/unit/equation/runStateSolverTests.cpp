#include <iostream>

#include <femx/eq/StateSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearStateSolver final : public equation::StateSolver
{
public:
  index_type numStates() const override
  {
    return 2;
  }

  index_type numParams() const override
  {
    return 2;
  }

  void solve(const Vector& params, Vector& state) override
  {
    if (state.size() != numStates())
    {
      state.resize(numStates());
    }
    else
    {
      state.setZero();
    }

    state[0] = 2.0 * params[0] + params[1];
    state[1] = -params[0] + 3.0 * params[1];
  }
};

class StateSolverTests : public TestBase
{
public:
  TestOutcome linearStateSolverOperations()
  {
    TestStatus status;
    status = true;

    LinearStateSolver solver;
    status *= (solver.numStates() == 2);
    status *= (solver.numParams() == 2);

    Vector params(2);
    params[0] = 0.25;
    params[1] = -0.5;

    Vector state;
    solver.solve(params, state);

    status *= (state.size() == solver.numStates());
    status *= isEqual(state[0], 2.0 * params[0] + params[1]);
    status *= isEqual(state[1], -params[0] + 3.0 * params[1]);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running state solver tests:\n";

  femx::tests::StateSolverTests test;

  femx::tests::TestingResults result;
  result += test.linearStateSolverOperations();

  return result.summary();
}
