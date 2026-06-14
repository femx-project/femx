#include <iostream>

#include <femx/inverse/AdjointSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearAdjointSolver final : public inverse::AdjointSolver
{
public:
  index_type numStates() const override
  {
    return 2;
  }

  index_type numParams() const override
  {
    return 1;
  }

  index_type numResiduals() const override
  {
    return 2;
  }

  void solve(const Vector& state,
             const Vector& params,
             const Vector& rhs,
             Vector&       adjoint) override
  {
    (void) state;
    (void) params;

    if (adjoint.size() != numResiduals())
    {
      adjoint.resize(numResiduals());
    }
    else
    {
      adjoint.setZero();
    }

    const real_type det = 2.0 * 11.0 - 7.0 * 3.0;
    adjoint[0]          = (11.0 * rhs[0] - 7.0 * rhs[1]) / det;
    adjoint[1]          = (-3.0 * rhs[0] + 2.0 * rhs[1]) / det;
  }
};

class AdjointSolverTests : public TestBase
{
public:
  TestOutcome linearAdjointSolverOperations()
  {
    TestStatus status;
    status = true;

    LinearAdjointSolver solver;
    status *= (solver.numStates() == 2);
    status *= (solver.numParams() == 1);
    status *= (solver.numResiduals() == 2);

    Vector state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector params(1);
    params[0] = 2.0;

    Vector rhs(2);
    rhs[0] = -1.5;
    rhs[1] = 4.0;

    Vector adjoint;
    solver.solve(state, params, rhs, adjoint);

    status *= (adjoint.size() == solver.numResiduals());
    status *= isEqual(2.0 * adjoint[0] + 7.0 * adjoint[1], rhs[0]);
    status *= isEqual(3.0 * adjoint[0] + 11.0 * adjoint[1], rhs[1]);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running adjoint solver tests:\n";

  femx::tests::AdjointSolverTests test;

  femx::tests::TestingResults result;
  result += test.linearAdjointSolverOperations();

  return result.summary();
}
