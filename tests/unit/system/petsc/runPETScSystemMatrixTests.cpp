#include <petscksp.h>

#include <cmath>
#include <iostream>

#include <femx/assembly/SystemAssembler.hpp>
#include <femx/equation/AssembledResidualEquation.hpp>
#include <femx/equation/MatrixNewtonStateSolver.hpp>
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

void resize(Vector& out, index_type size)
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
  : public equation::AssembledResidualEquation
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

  index_type numResiduals() const override
  {
    return 2;
  }

  void residual(const Vector& state,
                const Vector& params,
                Vector&       out) const override
  {
    resize(out, numResiduals());
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
    out.resize(numResiduals(), numStates());
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
    out.resize(numResiduals(), numParams());
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

    system::PETScSystemMatrix matrix;
    fillTestMatrix(matrix);

    Vector dir(2);
    dir[0] = 0.5;
    dir[1] = -1.0;

    Vector out;
    matrix.apply(dir, out);
    status *= isEqual(out[0], 1.0);
    status *= isEqual(out[1], -2.0);

    matrix.applyT(dir, out);
    status *= isEqual(out[0], 0.0);
    status *= isEqual(out[1], -2.5);

    return status.report(__func__);
  }

  TestOutcome kspSolvesMatrixAndTranspose()
  {
    TestStatus status;
    status = true;

    system::PETScSystemMatrix matrix;
    fillTestMatrix(matrix);

    system::KspLinearSolver solver;
    solver.options().pc_type     = PCJACOBI;
    solver.options().rtol        = 1.0e-12;
    solver.options().atol        = 1.0e-14;
    solver.options().use_opts_db = false;

    Vector rhs(2);
    rhs[0] = 1.0;
    rhs[1] = 2.0;

    Vector x;
    solver.solve(matrix, rhs, x);
    status *= (std::abs(4.0 * x[0] + x[1] - rhs[0]) < 1.0e-10);
    status *= (std::abs(2.0 * x[0] + 3.0 * x[1] - rhs[1]) < 1.0e-10);

    rhs[0] = 1.0;
    rhs[1] = 3.0;

    solver.solveT(matrix, rhs, x);
    status *= (std::abs(4.0 * x[0] + 2.0 * x[1] - rhs[0]) < 1.0e-10);
    status *= (std::abs(x[0] + 3.0 * x[1] - rhs[1]) < 1.0e-10);

    return status.report(__func__);
  }

  TestOutcome systemStateAndAdjointUsePETScSystemMatrix()
  {
    TestStatus status;
    status = true;

    LinearAssembledResidualEquation residual_equation;
    system::PETScSystemMatrix       state_jac;
    system::KspLinearSolver         lin_solver;
    lin_solver.options().pc_type     = PCJACOBI;
    lin_solver.options().rtol        = 1.0e-12;
    lin_solver.options().atol        = 1.0e-14;
    lin_solver.options().use_opts_db = false;

    equation::MatrixNewtonStateSolver state_solver(
        residual_equation, state_jac, lin_solver);
    inverse::MatrixEquationAdjointSolver adj_solver(
        residual_equation, state_jac, lin_solver);

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
    system::PETScSystemMatrix matrix;
    assembler.initMat(matrix);

    DenseMatrix local_matrix(space.numDofsPerElem(), space.numDofsPerElem());
    for (index_type i = 0; i < local_matrix.rows(); ++i)
    {
      for (index_type j = 0; j < local_matrix.cols(); ++j)
      {
        local_matrix(i, j) = 1.0 + static_cast<real_type>(i + j);
      }
    }

    assembler.addMat(0, local_matrix, matrix);
    matrix.finalize();

    Vector dir(space.numDofs());
    for (index_type i = 0; i < dir.size(); ++i)
    {
      dir[i] = 1.0;
    }

    Vector out;
    matrix.apply(dir, out);

    const auto dofs = space.elemDofs(0);
    for (index_type i = 0; i < local_matrix.rows(); ++i)
    {
      real_type expected = 0.0;
      for (index_type j = 0; j < local_matrix.cols(); ++j)
      {
        expected += local_matrix(i, j);
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
    system::PETScSystemVector vector;
    assembler.initVec(vector);

    Vector local_vector(space.numDofsPerElem());
    for (index_type i = 0; i < local_vector.size(); ++i)
    {
      local_vector[i] = 1.0;
    }
    assembler.addVec(0, local_vector, vector);

    for (index_type i = 0; i < local_vector.size(); ++i)
    {
      local_vector[i] = 10.0;
    }
    assembler.addVec(1, local_vector, vector);
    vector.finalize();

    Vector out;
    vector.copyToAll(out);
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
  static void fillTestMatrix(system::PETScSystemMatrix& matrix)
  {
    matrix.resize(2, 2);
    matrix.setZero();
    matrix.set(0, 0, 4.0);
    matrix.set(0, 1, 1.0);
    matrix.set(1, 0, 2.0);
    matrix.set(1, 1, 3.0);
    matrix.finalize();
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
