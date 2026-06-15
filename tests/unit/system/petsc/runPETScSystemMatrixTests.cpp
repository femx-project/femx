#include <petscksp.h>

#include <cmath>
#include <iostream>

#include <femx/assembly/SystemAssembler.hpp>
#include <femx/eq/AssembledNewtonStateSolver.hpp>
#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/inverse/MatrixEquationAdjointSolver.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>
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

  void assembleStateJac(const Vector&         state,
                        const Vector&         params,
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

  void assembleParamJac(const Vector&         state,
                        const Vector&         params,
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

class PETScSystemMatrixTests : public TestBase
{
public:
  TestOutcome appliesMatrixAndTranspose()
  {
    TestStatus status;
    status = true;

    system::PETScSystemMatrix mat;
    fillTestMatrix(mat);

    Vector dir(2);
    dir[0] = 0.5;
    dir[1] = -1.0;

    Vector out;
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

    system::PETScSystemMatrix mat;
    fillTestMatrix(mat);

    system::KspLinearSolver solver;
    solver.options().pc_type     = PCJACOBI;
    solver.options().rtol        = 1.0e-12;
    solver.options().atol        = 1.0e-14;
    solver.options().use_opts_db = false;

    Vector rhs(2);
    rhs[0] = 1.0;
    rhs[1] = 2.0;

    Vector x;
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

    LinearAssembledResidualEquation res_eq;
    system::PETScSystemMatrix       state_jac;
    system::KspLinearSolver         lin_solver;
    lin_solver.options().pc_type     = PCJACOBI;
    lin_solver.options().rtol        = 1.0e-12;
    lin_solver.options().atol        = 1.0e-14;
    lin_solver.options().use_opts_db = false;

    eq::AssembledNewtonStateSolver state_solver(
        res_eq, state_jac, lin_solver);
    inverse::MatrixEquationAdjointSolver adj_solver(
        res_eq, state_jac, lin_solver);

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

  TestOutcome assemblerScattersIntoPETScSystemMatrix()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    assembly::SystemAssembler assembler(space);
    system::PETScSystemMatrix mat;
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

    Vector dir(space.numDofs());
    for (Index i = 0; i < dir.size(); ++i)
    {
      dir[i] = 1.0;
    }

    Vector out;
    mat.apply(dir, out);

    const auto dofs = space.elemDofs(0);
    for (Index i = 0; i < local_mat.rows(); ++i)
    {
      Real expected = 0.0;
      for (Index j = 0; j < local_mat.cols(); ++j)
      {
        expected += local_mat(i, j);
      }
      status *= isEqual(out[dofs[static_cast<std::size_t>(i)]], expected);
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
    system::PETScSystemVector vec;
    assembler.initVec(vec);

    Vector local_vec(space.numDofsPerElem());
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

    Vector out;
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
  static void fillTestMatrix(system::PETScSystemMatrix& mat)
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
