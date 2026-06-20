#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <femx/common/Math.hpp>
#include <femx/common/Workspace.hpp>
#include <femx/eq/MatrixNewtonStateSolver.hpp>
#include <femx/eq/MatrixResidualEquation.hpp>
#include <femx/inverse/MatrixAdjointSolver.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/IndexSetList.hpp>
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

class LinearMatrixResidualEquation final
  : public eq::MatrixResidualEquation
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

  void assembleStateJac(const Vector<Real>&   state,
                        const Vector<Real>&   prm,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) prm;
    out.resize(numRes(), numStates());
    out.setZero();
    out.set(0, 0, 2.0);
    out.set(0, 1, 3.0);
    out.set(1, 0, 7.0);
    out.set(1, 1, 11.0);
  }

  void assembleParamJac(const Vector<Real>&   state,
                        const Vector<Real>&   prm,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) prm;
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

  TestOutcome rejectsOperator()
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

    LinearMatrixResidualEquation res_eq;
    auto                         pattern = makePattern();
    system::SparseSystemMatrix   state_jac(pattern);
    system::ReSolveLinearSolver  lin_solver(WorkspaceType::Cpu, options());

    eq::MatrixNewtonStateSolver state_solver(
        res_eq, state_jac, lin_solver);
    inverse::MatrixAdjointSolver adj_solver(
        res_eq, state_jac, lin_solver);

    Vector<Real> prm(2);
    prm[0] = 0.05;
    prm[1] = -0.02;

    Vector<Real> state;
    state_solver.solve(prm, state);
    status *= isEqual(state[0], -1.48);
    status *= isEqual(state[1], 0.89);

    Vector<Real> rhs(2);
    rhs[0] = -1.73;
    rhs[1] = 1.64;

    Vector<Real> adjoint;
    adj_solver.solve(state, prm, rhs, adjoint);
    status *= isEqual(2.0 * adjoint[0] + 7.0 * adjoint[1], rhs[0]);
    status *= isEqual(3.0 * adjoint[0] + 11.0 * adjoint[1], rhs[1]);

    return status.report(__func__);
  }

  TestOutcome reusesCpuIterativeTransposeSolver()
  {
    return repeatedIterativeTransposeSolves(WorkspaceType::Cpu, __func__);
  }

#if defined(RESOLVE_USE_CUDA)
  TestOutcome reusesCudaIterativeTransposeSolver()
  {
    return repeatedIterativeTransposeSolves(WorkspaceType::Cuda, __func__);
  }
#endif

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

  static system::ReSolveOptions iterativeOptions()
  {
    system::ReSolveOptions opts;
    opts.factor   = "none";
    opts.refactor = "none";
    opts.solve    = "fgmres";
    opts.precond  = "ilu0";
    opts.ir       = "none";
    opts.max_its  = 5000;
    opts.restart  = 200;
    opts.rtol     = 1.0e-8;
    opts.flexible = true;
    return opts;
  }

  static CsrPattern makePattern()
  {
    IndexSetList dofs;
    dofs.pushBack(Vector<Index>{0, 1});
    return CsrPattern(2, 2, dofs);
  }

  static CsrPattern makeDensePattern(Index n)
  {
    Vector<Index> dofs(n);
    for (Index i = 0; i < n; ++i)
    {
      dofs[i] = i;
    }
    IndexSetList elem_dofs;
    elem_dofs.pushBack(dofs);
    return CsrPattern(n, n, elem_dofs);
  }

  static void fillMatrix(system::SparseSystemMatrix& mat)
  {
    mat.set(0, 0, 2.0);
    mat.set(0, 1, 3.0);
    mat.set(1, 0, 7.0);
    mat.set(1, 1, 11.0);
    mat.finalize();
  }

  static void fillDenseMatrix(system::SparseSystemMatrix& mat, Real shift)
  {
    for (Index row = 0; row < mat.numRows(); ++row)
    {
      for (Index col = 0; col < mat.numCols(); ++col)
      {
        Real value = 0.0;
        if (row == col)
        {
          value = 4.0 + shift + 1.0e-3 * static_cast<Real>(row);
        }
        else
        {
          const Real dist = 1.0 + std::abs(row - col);
          const Real mag =
              1.0e-3
              * static_cast<Real>(1 + ((row + 3 * col) % 7))
              / dist;
          value = row < col ? mag : -0.5 * mag;
        }
        mat.set(row, col, value);
      }
    }
    mat.finalize();
  }

  static Real transposeRelativeResidual(const system::SparseSystemMatrix& mat,
                                        const Vector<Real>&               x,
                                        const Vector<Real>&               rhs)
  {
    Vector<Real> Ax;
    mat.applyT(x, Ax);
    for (Index i = 0; i < Ax.size(); ++i)
    {
      Ax[i] -= rhs[i];
    }
    return norm(Ax) / (1.0 + norm(rhs));
  }

  TestOutcome repeatedIterativeTransposeSolves(WorkspaceType work,
                                               const char*   name)
  {
    TestStatus status;
    status = true;

    constexpr Index             n       = 128;
    auto                        pattern = makeDensePattern(n);
    system::SparseSystemMatrix  mat(pattern);
    system::ReSolveLinearSolver solver(work, iterativeOptions());

    for (Index rep = 0; rep < 4; ++rep)
    {
      fillDenseMatrix(mat, 0.05 * static_cast<Real>(rep));

      Vector<Real> rhs(n);
      for (Index i = 0; i < n; ++i)
      {
        rhs[i] = std::cos(0.17 * static_cast<Real>(i + 1 + rep));
      }

      Vector<Real> x;
      solver.solveT(mat, rhs, x);
      status *= transposeRelativeResidual(mat, x, rhs) < 1.0e-6;
    }

    return status.report(name);
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
  result += test.rejectsOperator();
  result += test.worksInMatrixStateAndAdjointSolvers();
  result += test.reusesCpuIterativeTransposeSolver();
#if defined(RESOLVE_USE_CUDA)
  result += test.reusesCudaIterativeTransposeSolver();
#endif

  return result.summary();
}
