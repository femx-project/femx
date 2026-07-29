#include <petscmat.h>

#include "TestHelper.hpp"
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/PETScMatrix.hpp>

namespace femx::tests
{
namespace
{

bool rowReplacementKeepsNonzeroPattern(linalg::PETScMatrix& matrix)
{
  const HostVector<Index> rows{0, 1, 2};
  DenseMatrix             values(3, 3);
  for (Index i = 0; i < values.size(); ++i)
  {
    values.data()[i] = 1.0;
  }
  matrix.addBlock(rows.view(), rows.view(), values.view());
  matrix.finalize();

  MatInfo before{};
  if (MatGetInfo(matrix.mat(), MAT_LOCAL, &before) != PETSC_SUCCESS)
  {
    return false;
  }

  const HostVector<Index> constrained{1};
  matrix.replaceRows(constrained.view(), 1.0);

  MatInfo after{};
  return MatGetInfo(matrix.mat(), MAT_LOCAL, &after) == PETSC_SUCCESS
         && before.nz_used == 9.0 && after.nz_used == before.nz_used;
}

TestOutcome matricesKeepNonzeroPattern()
{
  TestStatus status(__func__);

  linalg::PETScMatrix default_matrix;
  default_matrix.resize(3, 3);
  status *= rowReplacementKeepsNonzeroPattern(default_matrix);

  const HostCsrPattern pattern{
      3,
      3,
      HostVector<Index>{0, 2, 5, 7},
      HostVector<Index>{0, 1, 0, 1, 2, 1, 2}};
  linalg::PETScMatrix preallocated_matrix;
  preallocated_matrix.resize(pattern);
  status *= rowReplacementKeepsNonzeroPattern(preallocated_matrix);

  return status.report();
}

} // namespace
} // namespace femx::tests

int main(int argc, char** argv)
{
  if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS)
  {
    return 1;
  }

  femx::tests::TestingResults results;
  results          += femx::tests::matricesKeepNonzeroPattern();
  const int status  = results.summary();

  if (PetscFinalize() != PETSC_SUCCESS)
  {
    return 1;
  }
  return status;
}
