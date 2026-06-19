#include <iostream>

#include <femx/eq/OperatorNewtonStateSolver.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/inverse/OperatorAdjointSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/DenseLinearSolver.hpp>
#include <femx/system/LinearOperator.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

void resize(Vector<Real>& out, Index size)
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

class TwoByTwoOperator final : public system::LinearOperator
{
public:
  Index numRows() const override
  {
    return 2;
  }

  Index numCols() const override
  {
    return 2;
  }

  void apply(const Vector<Real>& dir, Vector<Real>& out) const override
  {
    resize(out, numRows());
    out[0] = 2.0 * dir[0] + 3.0 * dir[1];
    out[1] = 7.0 * dir[0] + 11.0 * dir[1];
  }

  void applyT(const Vector<Real>& dir, Vector<Real>& out) const override
  {
    resize(out, numCols());
    out[0] = 2.0 * dir[0] + 7.0 * dir[1];
    out[1] = 3.0 * dir[0] + 11.0 * dir[1];
  }
};

class LinearResidualEquation final : public eq::ResidualEquation
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

  Index numRes() const override
  {
    return 2;
  }

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    resize(out, numRes());
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * prm[0] - 2.0 * prm[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * prm[0] + 4.0 * prm[1];
  }

  void applyStateJac(const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numRes());
    out[0] = 2.0 * dir[0] + 3.0 * dir[1];
    out[1] = 7.0 * dir[0] + 11.0 * dir[1];
  }

  void applyStateJacT(const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numStates());
    out[0] = 2.0 * lambda[0] + 7.0 * lambda[1];
    out[1] = 3.0 * lambda[0] + 11.0 * lambda[1];
  }

  void applyParamJac(const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numRes());
    out[0] = 5.0 * dir[0] - 2.0 * dir[1];
    out[1] = 13.0 * dir[0] + 4.0 * dir[1];
  }

  void applyParamJacT(const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numParams());
    out[0] = 5.0 * lambda[0] + 13.0 * lambda[1];
    out[1] = -2.0 * lambda[0] + 4.0 * lambda[1];
  }
};

class OperatorSolverTests : public TestBase
{
public:
  TestOutcome denseSolverSolvesOperatorAndTranspose()
  {
    TestStatus status;
    status = true;

    TwoByTwoOperator          op;
    system::DenseLinearSolver solver;

    Vector<Real> rhs(2);
    rhs[0] = 1.0;
    rhs[1] = 3.0;

    Vector<Real> x;
    solver.solve(op, rhs, x);
    status *= isEqual(2.0 * x[0] + 3.0 * x[1], rhs[0]);
    status *= isEqual(7.0 * x[0] + 11.0 * x[1], rhs[1]);

    solver.solveT(op, rhs, x);
    status *= isEqual(2.0 * x[0] + 7.0 * x[1], rhs[0]);
    status *= isEqual(3.0 * x[0] + 11.0 * x[1], rhs[1]);

    return status.report(__func__);
  }

  TestOutcome operatorNewtonStateSolverSolvesLinearResidual()
  {
    TestStatus status;
    status = true;

    LinearResidualEquation        res_eq;
    system::DenseLinearSolver     lin_solver;
    eq::OperatorNewtonStateSolver state_solver(res_eq, lin_solver);

    Vector<Real> prm(2);
    prm[0] = 0.05;
    prm[1] = -0.02;

    Vector<Real> state;
    state_solver.solve(prm, state);

    status *= isEqual(state[0], -1.48);
    status *= isEqual(state[1], 0.89);

    Vector<Real> res;
    res_eq.res(state, prm, res);
    status *= isEqual(res[0], 0.0);
    status *= isEqual(res[1], 0.0);

    return status.report(__func__);
  }

  TestOutcome operatorAdjointSolverSolvesTransposeJacobian()
  {
    TestStatus status;
    status = true;

    LinearResidualEquation         eq;
    system::DenseLinearSolver      lin_solver;
    inverse::OperatorAdjointSolver adj_solver(eq, lin_solver);

    Vector<Real> state(2);
    state[0] = -1.48;
    state[1] = 0.89;

    Vector<Real> prm(2);
    prm[0] = 0.05;
    prm[1] = -0.02;

    Vector<Real> rhs(2);
    rhs[0] = -1.73;
    rhs[1] = 1.64;

    Vector<Real> adjoint;
    adj_solver.solve(state, prm, rhs, adjoint);

    status *= isEqual(2.0 * adjoint[0] + 7.0 * adjoint[1], rhs[0]);
    status *= isEqual(3.0 * adjoint[0] + 11.0 * adjoint[1], rhs[1]);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running operator solver tests:\n";

  femx::tests::OperatorSolverTests test;

  femx::tests::TestingResults result;
  result += test.denseSolverSolvesOperatorAndTranspose();
  result += test.operatorNewtonStateSolverSolvesLinearResidual();
  result += test.operatorAdjointSolverSolvesTransposeJacobian();

  return result.summary();
}
