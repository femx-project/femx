#include <petscksp.h>

#include <cmath>
#include <iostream>

#include <femx/eq/MatrixFreeNewtonStateSolver.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/inverse/EquationAdjointSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearOperator.hpp>
#include <femx/system/petsc/KspLinearSolver.hpp>
#include <femx/system/petsc/PETScSystemMatrix.hpp>
#include <femx/system/petsc/PETScSystemVector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

void resize(Vector& out, Index size)
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

  void apply(const Vector& dir, Vector& out) const override
  {
    resize(out, numRows());
    out[0] = 2.0 * dir[0] + 3.0 * dir[1];
    out[1] = 7.0 * dir[0] + 11.0 * dir[1];
  }

  void applyT(const Vector& dir, Vector& out) const override
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

  void res(const Vector& state,
           const Vector& params,
           Vector&       out) const override
  {
    resize(out, numRes());
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * params[0] - 2.0 * params[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * params[0] + 4.0 * params[1];
  }

  void applyStateJac(const Vector& state,
                     const Vector& params,
                     const Vector& dir,
                     Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numRes());
    out[0] = 2.0 * dir[0] + 3.0 * dir[1];
    out[1] = 7.0 * dir[0] + 11.0 * dir[1];
  }

  void applyStateJacT(const Vector& state,
                      const Vector& params,
                      const Vector& lambda,
                      Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numStates());
    out[0] = 2.0 * lambda[0] + 7.0 * lambda[1];
    out[1] = 3.0 * lambda[0] + 11.0 * lambda[1];
  }

  void applyParamJac(const Vector& state,
                     const Vector& params,
                     const Vector& dir,
                     Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numRes());
    out[0] = 5.0 * dir[0] - 2.0 * dir[1];
    out[1] = 13.0 * dir[0] + 4.0 * dir[1];
  }

  void applyParamJacT(const Vector& state,
                      const Vector& params,
                      const Vector& lambda,
                      Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numParams());
    out[0] = 5.0 * lambda[0] + 13.0 * lambda[1];
    out[1] = -2.0 * lambda[0] + 4.0 * lambda[1];
  }
};

class KspLinearSolverTests : public TestBase
{
public:
  TestOutcome solveOperatorAndTranspose()
  {
    TestStatus status;
    status = true;

    TwoByTwoOperator        op;
    system::KspLinearSolver solver;
    solver.options().rtol        = 1.0e-12;
    solver.options().atol        = 1.0e-14;
    solver.options().use_opts_db = false;

    Vector rhs(2);
    rhs[0] = 1.0;
    rhs[1] = 3.0;

    Vector x;
    solver.solve(op, rhs, x);
    status *= (std::abs(2.0 * x[0] + 3.0 * x[1] - rhs[0]) < 1.0e-10);
    status *= (std::abs(7.0 * x[0] + 11.0 * x[1] - rhs[1]) < 1.0e-10);

    solver.solveT(op, rhs, x);
    status *= (std::abs(2.0 * x[0] + 7.0 * x[1] - rhs[0]) < 1.0e-10);
    status *= (std::abs(3.0 * x[0] + 11.0 * x[1] - rhs[1]) < 1.0e-10);

    return status.report(__func__);
  }

  TestOutcome solvesStateAndAdjointFromResidualEquation()
  {
    TestStatus status;
    status = true;

    LinearResidualEquation  equation;
    system::KspLinearSolver lin_solver;
    lin_solver.options().rtol        = 1.0e-12;
    lin_solver.options().atol        = 1.0e-14;
    lin_solver.options().use_opts_db = false;

    eq::MatrixFreeNewtonStateSolver state_solver(equation, lin_solver);
    inverse::EquationAdjointSolver  adj_solver(equation, lin_solver);

    Vector params(2);
    params[0] = 0.05;
    params[1] = -0.02;

    Vector state;
    state_solver.solve(params, state);
    status *= (std::abs(state[0] + 1.48) < 1.0e-10);
    status *= (std::abs(state[1] - 0.89) < 1.0e-10);

    Vector rhs(2);
    rhs[0] = -1.73;
    rhs[1] = 1.64;

    Vector adjoint;
    adj_solver.solve(state, params, rhs, adjoint);
    status *= (std::abs(2.0 * adjoint[0] + 7.0 * adjoint[1] - rhs[0])
               < 1.0e-10);
    status *= (std::abs(3.0 * adjoint[0] + 11.0 * adjoint[1] - rhs[1])
               < 1.0e-10);

    return status.report(__func__);
  }

  TestOutcome solvesPETScMatrixAndVectors()
  {
    TestStatus status;
    status = true;

    system::PETScSystemMatrix mat;
    mat.resize(2, 2);
    mat.setZero();
    mat.set(0, 0, 4.0);
    mat.set(0, 1, 1.0);
    mat.set(1, 0, 2.0);
    mat.set(1, 1, 3.0);
    mat.finalize();

    system::PETScSystemVector rhs;
    system::PETScSystemVector x;
    rhs.resize(2);
    x.resize(2);
    rhs.set(0, 1.0);
    rhs.set(1, 2.0);
    rhs.finalize();

    system::KspLinearSolver solver;
    solver.options().pc_type     = PCJACOBI;
    solver.options().rtol        = 1.0e-12;
    solver.options().atol        = 1.0e-14;
    solver.options().use_opts_db = false;

    solver.solve(mat, rhs, x);

    Vector out;
    x.copyToAll(out);
    status *= (std::abs(4.0 * out[0] + out[1] - 1.0) < 1.0e-10);
    status *= (std::abs(2.0 * out[0] + 3.0 * out[1] - 2.0) < 1.0e-10);
    status *= (solver.its() >= 0);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int argc, char** argv)
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (ierr != PETSC_SUCCESS)
  {
    return 1;
  }

  std::cout << "Running KSP linear system solver tests:\n";

  femx::tests::KspLinearSolverTests test;

  femx::tests::TestingResults result;
  result += test.solveOperatorAndTranspose();
  result += test.solvesStateAndAdjointFromResidualEquation();
  result += test.solvesPETScMatrixAndVectors();

  ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS)
  {
    return 1;
  }

  return result.summary();
}
