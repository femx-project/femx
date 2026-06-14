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

    FESpace vec_space(&mesh, &elem, 2);
    vec_space.setup();

    MixedFESpace mixed_space;
    mixed_space.addField(vec_space);
    mixed_space.addField(scalar_space);
    mixed_space.setup();

    assembly::DofLayout scalar_layout(scalar_space);
    status *= (scalar_layout.numElems() == 1);
    status *= (scalar_layout.numDofs() == 4);
    status *= (scalar_layout.numDofsPerElem() == 4);

    std::vector<Index> dofs;
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

    Vector global_vec;
    assembler.initVec(global_vec);

    Vector local_vec(4);
    for (Index i = 0; i < local_vec.size(); ++i)
    {
      local_vec[i] = 1.0;
    }
    assembler.addVec(0, local_vec, global_vec);

    for (Index i = 0; i < local_vec.size(); ++i)
    {
      local_vec[i] = 10.0;
    }
    assembler.addVec(1, local_vec, global_vec);

    status *= (global_vec.size() == 6);
    status *= isEqual(global_vec[0], 1.0);
    status *= isEqual(global_vec[1], 11.0);
    status *= isEqual(global_vec[2], 10.0);
    status *= isEqual(global_vec[3], 1.0);
    status *= isEqual(global_vec[4], 11.0);
    status *= isEqual(global_vec[5], 10.0);

    system::DenseSystemMatrix mat;
    DenseMatrix               expected(space.numDofs(), space.numDofs());
    DenseMatrix               local_mat(4, 4);
    assembler.initMat(mat);

    for (Index ic = 0; ic < space.numElems(); ++ic)
    {
      fillLocalMatrix(ic, local_mat);
      assembler.addMat(ic, local_mat, mat);

      const auto dofs = space.elemDofs(ic);
      for (Index i = 0; i < local_mat.rows(); ++i)
      {
        for (Index j = 0; j < local_mat.cols(); ++j)
        {
          expected(dofs[static_cast<std::size_t>(i)],
                   dofs[static_cast<std::size_t>(j)]) += local_mat(i, j);
        }
      }
    }

    for (Index i = 0; i < space.numDofs(); ++i)
    {
      for (Index j = 0; j < space.numDofs(); ++j)
      {
        status *= isEqual(mat.matrix()(i, j), expected(i, j));
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
    system::DenseSystemMatrix mat;
    assembler.initMat(mat);

    DenseMatrix local_mat(row_space.numDofsPerElem(),
                             col_space.numDofsPerElem());
    for (Index i = 0; i < local_mat.rows(); ++i)
    {
      for (Index j = 0; j < local_mat.cols(); ++j)
      {
        local_mat(i, j) = 10.0 * static_cast<Real>(i)
                             + static_cast<Real>(j);
      }
    }

    assembler.addMat(0, local_mat, mat);

    status *= (mat.numRows() == row_space.numDofs());
    status *= (mat.numCols() == col_space.numDofs());

    const auto row_dofs = row_space.elemDofs(0);
    const auto col_dofs = col_space.elemDofs(0);
    for (Index i = 0; i < local_mat.rows(); ++i)
    {
      for (Index j = 0; j < local_mat.cols(); ++j)
      {
        status *= isEqual(
            mat.matrix()(row_dofs[static_cast<std::size_t>(i)],
                         col_dofs[static_cast<std::size_t>(j)]),
            local_mat(i, j));
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
    system::DenseSystemVector global_vec;
    assembler.initVec(global_vec);

    Vector local_vec(space.numDofsPerElem());
    for (Index i = 0; i < local_vec.size(); ++i)
    {
      local_vec[i] = 1.0;
    }
    assembler.addVec(0, local_vec, global_vec);

    for (Index i = 0; i < local_vec.size(); ++i)
    {
      local_vec[i] = 10.0;
    }
    assembler.addVec(1, local_vec, global_vec);
    global_vec.finalize();

    const Vector& values  = global_vec.vector();
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
    system::SparseSystemMatrix mat(pattern);
    assembly::SystemAssembler  assembler(space);
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

  TestOutcome atomicModeScattersSparseMatrixAndVector()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    auto                       pattern = assembly::SparsityPatternBuilder::build(space);
    system::SparseSystemMatrix mat(pattern);
    assembly::SystemAssembler  assembler(
        space, assembly::AssemblyMode::Atomic);

    Vector global_vec;
    assembler.initVec(global_vec);
    assembler.initMat(mat);

    Vector      local_vec(space.numDofsPerElem());
    DenseMatrix local_mat(space.numDofsPerElem(), space.numDofsPerElem());
    for (Index i = 0; i < space.numDofsPerElem(); ++i)
    {
      local_vec[i] = 2.0;
      for (Index j = 0; j < space.numDofsPerElem(); ++j)
      {
        local_mat(i, j) = 1.0;
      }
    }

    assembler.addVec(0, local_vec, global_vec);
    assembler.addVec(0, local_vec, global_vec);
    assembler.addMat(0, local_mat, mat);
    assembler.addMat(0, local_mat, mat);
    mat.finalize();

    for (Index i = 0; i < global_vec.size(); ++i)
    {
      status *= isEqual(global_vec[i], 4.0);
    }

    Vector dir(space.numDofs());
    for (Index i = 0; i < dir.size(); ++i)
    {
      dir[i] = 1.0;
    }

    Vector out;
    mat.apply(dir, out);
    for (Index i = 0; i < out.size(); ++i)
    {
      status *= isEqual(out[i], 2.0 * static_cast<Real>(space.numDofsPerElem()));
    }

    return status.report(__func__);
  }

private:
  static void fillLocalMatrix(Index ic, DenseMatrix& local_mat)
  {
    for (Index i = 0; i < local_mat.rows(); ++i)
    {
      for (Index j = 0; j < local_mat.cols(); ++j)
      {
        local_mat(i, j) = 100.0 * static_cast<Real>(ic + 1)
                             + 10.0 * static_cast<Real>(i)
                             + static_cast<Real>(j);
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
