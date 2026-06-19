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
  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 1;
  }

  Index numRes() const override
  {
    return 2;
  }

  void solve(const Vector<Real>& state,
             const Vector<Real>& prm,
             const Vector<Real>& rhs,
             Vector<Real>&       adjoint) override
  {
    (void) state;
    (void) prm;

    if (adjoint.size() != numRes())
    {
      adjoint.resize(numRes());
    }
    else
    {
      adjoint.setZero();
    }

    const Real det = 2.0 * 11.0 - 7.0 * 3.0;
    adjoint[0]     = (11.0 * rhs[0] - 7.0 * rhs[1]) / det;
    adjoint[1]     = (-3.0 * rhs[0] + 2.0 * rhs[1]) / det;
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
    status *= (solver.numRes() == 2);

    Vector<Real> state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector<Real> prm(1);
    prm[0] = 2.0;

    Vector<Real> rhs(2);
    rhs[0] = -1.5;
    rhs[1] = 4.0;

    Vector<Real> adjoint;
    solver.solve(state, prm, rhs, adjoint);

    status *= (adjoint.size() == solver.numRes());
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
