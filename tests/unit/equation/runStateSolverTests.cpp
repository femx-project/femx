#include <iostream>

#include <femx/eq/StateSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearStateSolver final : public eq::StateSolver
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

  void solve(const Vector<Real>& prm, Vector<Real>& state) override
  {
    if (state.size() != numStates())
    {
      state.resize(numStates());
    }
    else
    {
      state.setZero();
    }

    state[0] = 2.0 * prm[0] + prm[1];
    state[1] = -prm[0] + 3.0 * prm[1];
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

    Vector<Real> prm(2);
    prm[0] = 0.25;
    prm[1] = -0.5;

    Vector<Real> state;
    solver.solve(prm, state);

    status *= (state.size() == solver.numStates());
    status *= isEqual(state[0], 2.0 * prm[0] + prm[1]);
    status *= isEqual(state[1], -prm[0] + 3.0 * prm[1]);

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
