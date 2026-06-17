#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <femx/common/Workspace.hpp>
#include <femx/eq/AssembledNewtonStateSolver.hpp>
#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/inverse/MatrixEquationAdjointSolver.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearOperator.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>
#include <femx/system/resolve/ReSolveLinearSolver.hpp>
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

class LinearAssembledResidualEquation final
  : public eq::AssembledResidualEquation
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
           const Vector<Real>& params,
           Vector<Real>&       out) const override
  {
    resize(out, numRes());
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * params[0] - 2.0 * params[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * params[0] + 4.0 * params[1];
  }

  void assembleStateJac(const Vector<Real>&   state,
                        const Vector<Real>&   params,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) params;
    out.resize(numRes(), numStates());
    out.setZero();
    out.set(0, 0, 2.0);
    out.set(0, 1, 3.0);
    out.set(1, 0, 7.0);
    out.set(1, 1, 11.0);
  }

  void assembleParamJac(const Vector<Real>&   state,
                        const Vector<Real>&   params,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) params;
    out.resize(numRes(), numParams());
    out.setZero();
    out.set(0, 0, 5.0);
    out.set(0, 1, -2.0);
    out.set(1, 0, 13.0);
    out.set(1, 1, 4.0);
  }
};

class ReSolveLinearSolverTests : public TestBase
{
public:
  TestOutcome solveSparseSystemMatrixAndTranspose()
  {
    TestStatus status;
    status = true;

    auto                       pattern = makePattern();
    system::SparseSystemMatrix mat(pattern);
    fillMatrix(mat);

    system::ReSolveLinearSolver solver(WorkspaceType::Cpu, options());

    Vector<Real> rhs(2);
    rhs[0] = 1.0;
    rhs[1] = 3.0;

    Vector<Real> x;
    solver.solve(mat, rhs, x);
    status *= isEqual(2.0 * x[0] + 3.0 * x[1], rhs[0]);
    status *= isEqual(7.0 * x[0] + 11.0 * x[1], rhs[1]);

    solver.solveT(mat, rhs, x);
    status *= isEqual(2.0 * x[0] + 7.0 * x[1], rhs[0]);
    status *= isEqual(3.0 * x[0] + 11.0 * x[1], rhs[1]);

    return status.report(__func__);
  }

  TestOutcome preservesDirectSparseMatrixSolveApi()
  {
    TestStatus status;
    status = true;

    auto                       pattern = makePattern();
    system::SparseSystemMatrix mat(pattern);
    fillMatrix(mat);

    system::ReSolveLinearSolver solver(WorkspaceType::Cpu, options());
    Vector<Real>                rhs(2);
    rhs[0] = 1.0;
    rhs[1] = 3.0;

    Vector<Real> x;
    solver.setOperator(mat.matrix());
    solver.solve(rhs, x);
    status *= isEqual(2.0 * x[0] + 3.0 * x[1], rhs[0]);
    status *= isEqual(7.0 * x[0] + 11.0 * x[1], rhs[1]);

    return status.report(__func__);
  }

  TestOutcome rejectsMatrixFreeOperator()
  {
    TestStatus status;
    status = true;

    TwoByTwoOperator            op;
    system::ReSolveLinearSolver solver(WorkspaceType::Cpu, options());

    Vector<Real> rhs(2);
    rhs[0] = 1.0;
    rhs[1] = 3.0;
    Vector<Real> x;

    bool threw = false;
    try
    {
      solver.solve(op, rhs, x);
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }
    status *= threw;

    return status.report(__func__);
  }

  TestOutcome worksInMatrixStateAndAdjointSolvers()
  {
    TestStatus status;
    status = true;

    LinearAssembledResidualEquation res_eq;
    auto                            pattern = makePattern();
    system::SparseSystemMatrix      state_jac(pattern);
    system::ReSolveLinearSolver     lin_solver(WorkspaceType::Cpu, options());

    eq::AssembledNewtonStateSolver state_solver(
        res_eq, state_jac, lin_solver);
    inverse::MatrixEquationAdjointSolver adj_solver(
        res_eq, state_jac, lin_solver);

    Vector<Real> params(2);
    params[0] = 0.05;
    params[1] = -0.02;

    Vector<Real> state;
    state_solver.solve(params, state);
    status *= isEqual(state[0], -1.48);
    status *= isEqual(state[1], 0.89);

    Vector<Real> rhs(2);
    rhs[0] = -1.73;
    rhs[1] = 1.64;

    Vector<Real> adjoint;
    adj_solver.solve(state, params, rhs, adjoint);
    status *= isEqual(2.0 * adjoint[0] + 7.0 * adjoint[1], rhs[0]);
    status *= isEqual(3.0 * adjoint[0] + 11.0 * adjoint[1], rhs[1]);

    return status.report(__func__);
  }

private:
  static system::ReSolveOptions options()
  {
    system::ReSolveOptions opts;
    opts.factor   = "klu";
    opts.refactor = "none";
    opts.solve    = "klu";
    opts.precond  = "none";
    opts.ir       = "none";
    return opts;
  }

  static CsrPattern makePattern()
  {
    return CsrPattern(2, 2, std::vector<Vector<Index>>{{0, 1}});
  }

  static void fillMatrix(system::SparseSystemMatrix& mat)
  {
    mat.set(0, 0, 2.0);
    mat.set(0, 1, 3.0);
    mat.set(1, 0, 7.0);
    mat.set(1, 1, 11.0);
    mat.finalize();
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running ReSolve linear solver tests:\n";

  femx::tests::ReSolveLinearSolverTests test;

  femx::tests::TestingResults result;
  result += test.solveSparseSystemMatrixAndTranspose();
  result += test.preservesDirectSparseMatrixSolveApi();
  result += test.rejectsMatrixFreeOperator();
  result += test.worksInMatrixStateAndAdjointSolvers();

  return result.summary();
}
