#include <petscksp.h>

#include <cmath>
#include <iostream>

#include <femx/assembly/SystemAssembler.hpp>
#include <femx/solve/MatrixNewtonStateSolver.hpp>
#include <femx/problem/MatrixResidualEquation.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/solve/MatrixAdjointSolver.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/algebra/backends/petsc/KspLinearSolver.hpp>
#include <femx/algebra/backends/petsc/PETScSystemMatrix.hpp>
#include <femx/algebra/backends/petsc/PETScSystemVector.hpp>
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

class LinearMatrixResidualEquation final
  : public problem::MatrixResidualEquation
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
                        algebra::SystemMatrix& out) const override
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
                        algebra::SystemMatrix& out) const override
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

class PETScSystemMatrixTests : public TestBase
{
public:
  TestOutcome appliesMatrixAndTranspose()
  {
    TestStatus status;
    status = true;

    algebra::PETScSystemMatrix mat;
    fillTestMatrix(mat);

    Vector<Real> dir(2);
    dir[0] = 0.5;
    dir[1] = -1.0;

    Vector<Real> out;
    mat.apply(dir, out);
    status *= isEqual(out[0], 1.0);
    status *= isEqual(out[1], -2.0);

    mat.applyT(dir, out);
    status *= isEqual(out[0], 0.0);
    status *= isEqual(out[1], -2.5);

    return status.report(__func__);
  }

  TestOutcome kspSolvesMatrixAndTranspose()
  {
    TestStatus status;
    status = true;

    algebra::PETScSystemMatrix mat;
    fillTestMatrix(mat);

    algebra::KspLinearSolver solver;
    solver.options().pc_type     = PCJACOBI;
    solver.options().rtol        = 1.0e-12;
    solver.options().atol        = 1.0e-14;
    solver.options().use_opts_db = false;

    Vector<Real> rhs(2);
    rhs[0] = 1.0;
    rhs[1] = 2.0;

    Vector<Real> x;
    solver.solve(mat, rhs, x);
    status *= (std::abs(4.0 * x[0] + x[1] - rhs[0]) < 1.0e-10);
    status *= (std::abs(2.0 * x[0] + 3.0 * x[1] - rhs[1]) < 1.0e-10);

    rhs[0] = 1.0;
    rhs[1] = 3.0;

    solver.solveT(mat, rhs, x);
    status *= (std::abs(4.0 * x[0] + 2.0 * x[1] - rhs[0]) < 1.0e-10);
    status *= (std::abs(x[0] + 3.0 * x[1] - rhs[1]) < 1.0e-10);

    return status.report(__func__);
  }

  TestOutcome systemStateAndAdjointUsePETScSystemMatrix()
  {
    TestStatus status;
    status = true;

    LinearMatrixResidualEquation res_eq;
    algebra::PETScSystemMatrix    state_jac;
    algebra::KspLinearSolver      lin_solver;
    lin_solver.options().pc_type     = PCJACOBI;
    lin_solver.options().rtol        = 1.0e-12;
    lin_solver.options().atol        = 1.0e-14;
    lin_solver.options().use_opts_db = false;

    solve::MatrixNewtonStateSolver state_solver(
        res_eq, state_jac, lin_solver);
    solve::MatrixAdjointSolver adj_solver(
        res_eq, state_jac, lin_solver);

    Vector<Real> prm(2);
    prm[0] = 0.05;
    prm[1] = -0.02;

    Vector<Real> state;
    state_solver.solve(prm, state);
    status *= (std::abs(state[0] + 1.48) < 1.0e-10);
    status *= (std::abs(state[1] - 0.89) < 1.0e-10);

    Vector<Real> rhs(2);
    rhs[0] = -1.73;
    rhs[1] = 1.64;

    Vector<Real> adjoint;
    adj_solver.solve(state, prm, rhs, adjoint);
    status *= (std::abs(2.0 * adjoint[0] + 7.0 * adjoint[1] - rhs[0])
               < 1.0e-10);
    status *= (std::abs(3.0 * adjoint[0] + 11.0 * adjoint[1] - rhs[1])
               < 1.0e-10);

    return status.report(__func__);
  }

  TestOutcome assemblerScattersIntoPETScSystemMatrix()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    assembly::SystemAssembler assembler(space);
    algebra::PETScSystemMatrix mat;
    assembler.initMat(mat);

    DenseMatrix local_mat(space.numDofsPerElem(), space.numDofsPerElem());
    for (Index i = 0; i < local_mat.rows(); ++i)
    {
      for (Index j = 0; j < local_mat.cols(); ++j)
      {
        local_mat(i, j) = 1.0 + static_cast<Real>(i + j);
      }
    }

    assembler.addMat(0, local_mat, mat);
    mat.finalize();

    Vector<Real> dir(space.numDofs());
    for (Index i = 0; i < dir.size(); ++i)
    {
      dir[i] = 1.0;
    }

    Vector<Real> out;
    mat.apply(dir, out);

    const auto dofs = space.elemDofs(0);
    for (Index i = 0; i < local_mat.rows(); ++i)
    {
      Real expected = 0.0;
      for (Index j = 0; j < local_mat.cols(); ++j)
      {
        expected += local_mat(i, j);
      }
      status *= isEqual(out[dofs[i]], expected);
    }

    return status.report(__func__);
  }

  TestOutcome assemblerScattersIntoPETScSystemVector()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(2, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    assembly::SystemAssembler assembler(space);
    algebra::PETScSystemVector vec;
    assembler.initVec(vec);

    Vector<Real> local_vec(space.numDofsPerElem());
    for (Index i = 0; i < local_vec.size(); ++i)
    {
      local_vec[i] = 1.0;
    }
    assembler.addVec(0, local_vec, vec);

    for (Index i = 0; i < local_vec.size(); ++i)
    {
      local_vec[i] = 10.0;
    }
    assembler.addVec(1, local_vec, vec);
    vec.finalize();

    Vector<Real> out;
    vec.copyToAll(out);
    status *= (out.size() == 6);
    status *= isEqual(out[0], 1.0);
    status *= isEqual(out[1], 11.0);
    status *= isEqual(out[2], 10.0);
    status *= isEqual(out[3], 1.0);
    status *= isEqual(out[4], 11.0);
    status *= isEqual(out[5], 10.0);

    return status.report(__func__);
  }

private:
  static void fillTestMatrix(algebra::PETScSystemMatrix& mat)
  {
    mat.resize(2, 2);
    mat.setZero();
    mat.set(0, 0, 4.0);
    mat.set(0, 1, 1.0);
    mat.set(1, 0, 2.0);
    mat.set(1, 1, 3.0);
    mat.finalize();
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

  std::cout << "Running PETSc system matrix tests:\n";

  femx::tests::PETScSystemMatrixTests test;

  femx::tests::TestingResults result;
  result += test.appliesMatrixAndTranspose();
  result += test.kspSolvesMatrixAndTranspose();
  result += test.systemStateAndAdjointUsePETScSystemMatrix();
  result += test.assemblerScattersIntoPETScSystemMatrix();
  result += test.assemblerScattersIntoPETScSystemVector();

  ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS)
  {
    return 1;
  }

  return result.summary();
}
