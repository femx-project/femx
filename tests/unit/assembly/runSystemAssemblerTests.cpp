#include <iostream>
#include <vector>

#include <femx/assembly/DofLayout.hpp>
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/assembly/SystemAssembler.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/native/DenseSystemMatrix.hpp>
#include <femx/system/native/DenseSystemVector.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class SystemAssemblerTests : public TestBase
{
public:
  TestOutcome dofLayoutViewsScalarAndMixedSpaces()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 elem;

    FESpace scalar_space(&mesh, &elem);
    scalar_space.setup();

    FESpace vector_space(&mesh, &elem, 2);
    vector_space.setup();

    MixedFESpace mixed_space;
    mixed_space.addField(vector_space);
    mixed_space.addField(scalar_space);
    mixed_space.setup();

    assembly::DofLayout scalar_layout(scalar_space);
    status *= (scalar_layout.numElems() == 1);
    status *= (scalar_layout.numDofs() == 4);
    status *= (scalar_layout.numDofsPerElem() == 4);

    std::vector<index_type> dofs;
    scalar_layout.elemDofs(0, dofs);
    status *= (dofs.size() == 4);
    status *= (dofs[0] == 0);
    status *= (dofs[1] == 1);
    status *= (dofs[2] == 3);
    status *= (dofs[3] == 2);

    assembly::DofLayout mixed_layout(mixed_space);
    status *= (mixed_layout.numElems() == 1);
    status *= (mixed_layout.numDofs() == 12);
    status *= (mixed_layout.numDofsPerElem() == 12);

    mixed_layout.elemDofs(0, dofs);
    status *= (dofs.size() == 12);
    status *= (dofs[0] == 0);
    status *= (dofs[1] == 1);
    status *= (dofs[2] == 2);
    status *= (dofs[3] == 3);
    status *= (dofs[4] == 6);
    status *= (dofs[5] == 7);
    status *= (dofs[6] == 4);
    status *= (dofs[7] == 5);
    status *= (dofs[8] == 8);
    status *= (dofs[9] == 9);
    status *= (dofs[10] == 11);
    status *= (dofs[11] == 10);

    return status.report(__func__);
  }

  TestOutcome scattersVectorAndDenseMatrix()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(2, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    assembly::SystemAssembler assembler(space);

    Vector global_vector;
    assembler.initVec(global_vector);

    Vector local_vector(4);
    for (index_type i = 0; i < local_vector.size(); ++i)
    {
      local_vector[i] = 1.0;
    }
    assembler.addVec(0, local_vector, global_vector);

    for (index_type i = 0; i < local_vector.size(); ++i)
    {
      local_vector[i] = 10.0;
    }
    assembler.addVec(1, local_vector, global_vector);

    status *= (global_vector.size() == 6);
    status *= isEqual(global_vector[0], 1.0);
    status *= isEqual(global_vector[1], 11.0);
    status *= isEqual(global_vector[2], 10.0);
    status *= isEqual(global_vector[3], 1.0);
    status *= isEqual(global_vector[4], 11.0);
    status *= isEqual(global_vector[5], 10.0);

    system::DenseSystemMatrix matrix;
    DenseMatrix               expected(space.numDofs(), space.numDofs());
    DenseMatrix               local_matrix(4, 4);
    assembler.initMat(matrix);

    for (index_type cell = 0; cell < space.numElems(); ++cell)
    {
      fillLocalMatrix(cell, local_matrix);
      assembler.addMat(cell, local_matrix, matrix);

      const auto dofs = space.elemDofs(cell);
      for (index_type i = 0; i < local_matrix.rows(); ++i)
      {
        for (index_type j = 0; j < local_matrix.cols(); ++j)
        {
          expected(dofs[static_cast<std::size_t>(i)],
                   dofs[static_cast<std::size_t>(j)]) += local_matrix(i, j);
        }
      }
    }

    for (index_type i = 0; i < space.numDofs(); ++i)
    {
      for (index_type j = 0; j < space.numDofs(); ++j)
      {
        status *= isEqual(matrix.matrix()(i, j), expected(i, j));
      }
    }

    return status.report(__func__);
  }

  TestOutcome scattersRectangularDenseMatrix()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 elem;

    FESpace row_space(&mesh, &elem);
    row_space.setup();

    FESpace col_space(&mesh, &elem, 2);
    col_space.setup();

    assembly::SystemAssembler assembler(row_space, col_space);
    system::DenseSystemMatrix matrix;
    assembler.initMat(matrix);

    DenseMatrix local_matrix(row_space.numDofsPerElem(),
                             col_space.numDofsPerElem());
    for (index_type i = 0; i < local_matrix.rows(); ++i)
    {
      for (index_type j = 0; j < local_matrix.cols(); ++j)
      {
        local_matrix(i, j) = 10.0 * static_cast<real_type>(i)
                             + static_cast<real_type>(j);
      }
    }

    assembler.addMat(0, local_matrix, matrix);

    status *= (matrix.numRows() == row_space.numDofs());
    status *= (matrix.numCols() == col_space.numDofs());

    const auto row_dofs = row_space.elemDofs(0);
    const auto col_dofs = col_space.elemDofs(0);
    for (index_type i = 0; i < local_matrix.rows(); ++i)
    {
      for (index_type j = 0; j < local_matrix.cols(); ++j)
      {
        status *= isEqual(
            matrix.matrix()(row_dofs[static_cast<std::size_t>(i)],
                            col_dofs[static_cast<std::size_t>(j)]),
            local_matrix(i, j));
      }
    }

    return status.report(__func__);
  }

  TestOutcome scattersIntoDenseSystemVector()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(2, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    assembly::SystemAssembler assembler(space);
    system::DenseSystemVector global_vector;
    assembler.initVec(global_vector);

    Vector local_vector(space.numDofsPerElem());
    for (index_type i = 0; i < local_vector.size(); ++i)
    {
      local_vector[i] = 1.0;
    }
    assembler.addVec(0, local_vector, global_vector);

    for (index_type i = 0; i < local_vector.size(); ++i)
    {
      local_vector[i] = 10.0;
    }
    assembler.addVec(1, local_vector, global_vector);
    global_vector.finalize();

    const Vector& values  = global_vector.vector();
    status               *= (values.size() == 6);
    status               *= isEqual(values[0], 1.0);
    status               *= isEqual(values[1], 11.0);
    status               *= isEqual(values[2], 10.0);
    status               *= isEqual(values[3], 1.0);
    status               *= isEqual(values[4], 11.0);
    status               *= isEqual(values[5], 10.0);

    return status.report(__func__);
  }

  TestOutcome scattersSparseMatrixAndAppliesIt()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    auto                       pattern = assembly::SparsityPatternBuilder::build(space);
    system::SparseSystemMatrix matrix(pattern);
    assembly::SystemAssembler  assembler(space);
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

  TestOutcome atomicModeScattersSparseMatrixAndVector()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    auto                       pattern = assembly::SparsityPatternBuilder::build(space);
    system::SparseSystemMatrix matrix(pattern);
    assembly::SystemAssembler  assembler(
        space, assembly::SystemAssembler::AssemblyMode::Atomic);

    Vector global_vector;
    assembler.initVec(global_vector);
    assembler.initMat(matrix);

    Vector      local_vector(space.numDofsPerElem());
    DenseMatrix local_matrix(space.numDofsPerElem(), space.numDofsPerElem());
    for (index_type i = 0; i < space.numDofsPerElem(); ++i)
    {
      local_vector[i] = 2.0;
      for (index_type j = 0; j < space.numDofsPerElem(); ++j)
      {
        local_matrix(i, j) = 1.0;
      }
    }

    assembler.addVec(0, local_vector, global_vector);
    assembler.addVec(0, local_vector, global_vector);
    assembler.addMat(0, local_matrix, matrix);
    assembler.addMat(0, local_matrix, matrix);
    matrix.finalize();

    for (index_type i = 0; i < global_vector.size(); ++i)
    {
      status *= isEqual(global_vector[i], 4.0);
    }

    Vector dir(space.numDofs());
    for (index_type i = 0; i < dir.size(); ++i)
    {
      dir[i] = 1.0;
    }

    Vector out;
    matrix.apply(dir, out);
    for (index_type i = 0; i < out.size(); ++i)
    {
      status *= isEqual(out[i], 2.0 * static_cast<real_type>(space.numDofsPerElem()));
    }

    return status.report(__func__);
  }

private:
  static void fillLocalMatrix(index_type cell, DenseMatrix& local_matrix)
  {
    for (index_type i = 0; i < local_matrix.rows(); ++i)
    {
      for (index_type j = 0; j < local_matrix.cols(); ++j)
      {
        local_matrix(i, j) = 100.0 * static_cast<real_type>(cell + 1)
                             + 10.0 * static_cast<real_type>(i)
                             + static_cast<real_type>(j);
      }
    }
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running system assembler tests:\n";

  femx::tests::SystemAssemblerTests test;

  femx::tests::TestingResults result;
  result += test.dofLayoutViewsScalarAndMixedSpaces();
  result += test.scattersVectorAndDenseMatrix();
  result += test.scattersRectangularDenseMatrix();
  result += test.scattersIntoDenseSystemVector();
  result += test.scattersSparseMatrixAndAppliesIt();
  result += test.atomicModeScattersSparseMatrixAndVector();

  return result.summary();
}
