#include <petscmat.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "TestHelper.hpp"
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/MpiContext.hpp>
#include <femx/linalg/petsc/PETScLinearSolver.hpp>
#include <femx/linalg/petsc/PETScMatrix.hpp>
#include <femx/linalg/petsc/PETScPartition.hpp>

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

HostCsrPattern interleavedPathPattern()
{
  const HostVector<Index> physical_to_app{
      0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};
  const Index size = physical_to_app.size();

  std::vector<HostVector<Index>> rows(
      static_cast<std::size_t>(size));
  for (Index physical = 0; physical < size; ++physical)
  {
    const Index row = physical_to_app[physical];
    rows[row].push_back(row);
    if (physical > 0)
    {
      rows[row].push_back(physical_to_app[physical - 1]);
    }
    if (physical + 1 < size)
    {
      rows[row].push_back(physical_to_app[physical + 1]);
    }
  }

  HostVector<Index> row_ptr(size + 1, 0);
  HostVector<Index> col_ind;
  for (Index row = 0; row < size; ++row)
  {
    auto& columns = rows[row];
    std::sort(columns.begin(), columns.end());
    for (const Index column : columns)
    {
      col_ind.push_back(column);
    }
    row_ptr[row + 1] = col_ind.size();
  }
  return {size, size, std::move(row_ptr), std::move(col_ind)};
}

TestOutcome partitionedSolveKeepsApplicationOrdering()
{
  TestStatus status(__func__);

  PetscMPIInt comm_size  = 1;
  PetscMPIInt rank       = 0;
  status                *= MPI_Comm_size(PETSC_COMM_WORLD, &comm_size) == MPI_SUCCESS;
  status                *= MPI_Comm_rank(PETSC_COMM_WORLD, &rank) == MPI_SUCCESS;
  if (comm_size < 2)
  {
    status.skipTest();
    return status.report();
  }

  const HostCsrPattern pattern = interleavedPathPattern();
  linalg::PETScMatrix  matrix(PETSC_COMM_WORLD);
  matrix.resize(pattern);
  const auto partition  = matrix.partition();
  status               *= partition != nullptr;
  if (!partition)
  {
    return status.report();
  }

  linalg::MpiContext context(PETSC_COMM_WORLD);
  context.setPartition(partition);
  const linalg::IndexRange candidate_range =
      context.elementRange(pattern.rows());
  status *= candidate_range.begin == 0
            && candidate_range.end == pattern.rows();
  for (Index element = 0; element < pattern.rows(); ++element)
  {
    const HostVector<Index> rows{element};
    int                     local_owner =
        context.ownsElement(element, pattern.rows(), rows.view()) ? 1 : 0;
    int num_owners  = 0;
    status         *= MPI_Allreduce(&local_owner,
                            &num_owners,
                            1,
                            MPI_INT,
                            MPI_SUM,
                            PETSC_COMM_WORLD)
              == MPI_SUCCESS;
    status *= num_owners == 1;
  }

#if defined(PETSC_HAVE_PARMETIS)
  status *= partition->type() == MATPARTITIONINGPARMETIS;
#elif defined(PETSC_HAVE_PTSCOTCH)
  status *= partition->type() == MATPARTITIONINGPTSCOTCH;
#endif

  HostVector<Real> rhs(pattern.rows(), 0.0);
  for (Index row = 0; row < pattern.rows(); ++row)
  {
    const PetscInt petsc_row  = partition->petscIndex(row);
    status                   *= partition->applicationIndex(petsc_row) == row;

    const bool owned  = partition->owner(row) == rank;
    status           *= owned
              == (petsc_row >= partition->begin()
                  && petsc_row < partition->end());
    if (!owned)
    {
      continue;
    }

    rhs[row] = 2.0;
    for (Index k = pattern.rowPtr()[row];
         k < pattern.rowPtr()[row + 1];
         ++k)
    {
      const Index column = pattern.colInd()[k];
      const Real  value  = column == row ? 2.0 : -1.0;
      matrix.set(row, column, value);
      if (column != row)
      {
        rhs[row] -= 1.0;
      }
    }
  }
  matrix.finalize();

  linalg::PETScLinearSolver solver(PETSC_COMM_WORLD);
  solver.opts().type = KSPCG;
  HostVector<Real> solution;
  solver.solve(matrix, rhs, solution);
  status *= solution.size() == pattern.rows();
  for (const Real value : solution)
  {
    status *= std::abs(value - 1.0) < 1.0e-7;
  }

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
  results          += femx::tests::partitionedSolveKeepsApplicationOrdering();
  const int status  = results.summary();

  if (PetscFinalize() != PETSC_SUCCESS)
  {
    return 1;
  }
  return status;
}
