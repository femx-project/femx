#include <stdexcept>
#include <string>

#include <femx/linalg/petsc/PETScAssemblyMatrix.hpp>
#include <femx/linalg/petsc/VectorConversion.hpp>

namespace femx
{
namespace linalg
{

PETScAssemblyMatrix::PETScAssemblyMatrix(MPI_Comm comm)
  : comm_(comm)
{
}

PETScAssemblyMatrix::~PETScAssemblyMatrix()
{
  if (mat_ != nullptr)
  {
    MatDestroy(&mat_);
  }
}

Index PETScAssemblyMatrix::numRows() const
{
  return rows_;
}

Index PETScAssemblyMatrix::numCols() const
{
  return cols_;
}

Mat PETScAssemblyMatrix::mat() const
{
  if (mat_ == nullptr)
  {
    throw std::runtime_error("PETScAssemblyMatrix is not initialized");
  }
  return mat_;
}

MPI_Comm PETScAssemblyMatrix::comm() const
{
  return comm_;
}

void PETScAssemblyMatrix::setDefaultNonzerosPerRow(Index count)
{
  if (count <= 0)
  {
    throw std::runtime_error("PETScAssemblyMatrix preallocation must be positive");
  }
  default_nonzeros_per_row_ = count;
}

void PETScAssemblyMatrix::resize(Index rows, Index cols)
{
  checkInitialized();

  if (mat_ != nullptr && rows_ == rows && cols_ == cols)
  {
    setZero();
    return;
  }

  if (mat_ != nullptr)
  {
    check(MatDestroy(&mat_), "MatDestroy");
  }

  rows_ = rows;
  cols_ = cols;

  const PetscInt petsc_rows = static_cast<PetscInt>(rows_);
  const PetscInt petsc_cols = static_cast<PetscInt>(cols_);

  PetscMPIInt comm_size = 1;
  checkMPI(MPI_Comm_size(comm_, &comm_size), "MPI_Comm_size");

  const PetscInt local_rows = comm_size == 1 ? petsc_rows : PETSC_DECIDE;
  const PetscInt local_cols = comm_size == 1 ? petsc_cols : PETSC_DECIDE;

  Index nonzeros_per_row = cols_ > 0 ? default_nonzeros_per_row_ : 1;
  if (cols_ > 0 && cols_ < nonzeros_per_row)
  {
    nonzeros_per_row = cols_;
  }

  check(MatCreateAIJ(comm_,
                     local_rows,
                     local_cols,
                     petsc_rows,
                     petsc_cols,
                     static_cast<PetscInt>(nonzeros_per_row),
                     nullptr,
                     static_cast<PetscInt>(nonzeros_per_row),
                     nullptr,
                     &mat_),
        "MatCreateAIJ");
  check(MatSetOption(mat_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE),
        "MatSetOption");
}

void PETScAssemblyMatrix::resize(const CsrPattern& pattern, const PETScVector& lyt)
{
  checkInitialized();

  if (lyt.size() != pattern.rows())
  {
    throw std::runtime_error(
        "PETScAssemblyMatrix row layout does not match sparsity pattern");
  }

  if (mat_ != nullptr && rows_ == pattern.rows() && cols_ == pattern.cols())
  {
    setZero();
    return;
  }

  if (mat_ != nullptr)
  {
    check(MatDestroy(&mat_), "MatDestroy");
  }

  comm_ = lyt.comm();
  rows_ = pattern.rows();
  cols_ = pattern.cols();

  const PetscInt begin = lyt.ownershipBegin();
  const PetscInt end   = lyt.ownershipEnd();

  Vector<PetscInt> d_nnz;
  Vector<PetscInt> o_nnz;
  computePreallocation(pattern, begin, end, d_nnz, o_nnz);

  check(MatCreateAIJ(comm_,
                     end - begin,
                     PETSC_DECIDE,
                     static_cast<PetscInt>(rows_),
                     static_cast<PetscInt>(cols_),
                     0,
                     d_nnz.empty() ? nullptr : d_nnz.data(),
                     0,
                     o_nnz.empty() ? nullptr : o_nnz.data(),
                     &mat_),
        "MatCreateAIJ");

  check(MatSetOption(mat_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE),
        "MatSetOption");
  check(MatSetUp(mat_), "MatSetUp");
}

void PETScAssemblyMatrix::setZero()
{
  if (mat_ == nullptr)
  {
    return;
  }
  check(MatZeroEntries(mat_), "MatZeroEntries");
}

void PETScAssemblyMatrix::set(Index row, Index col, Real value)
{
  setValue(row, col, value, INSERT_VALUES);
}

void PETScAssemblyMatrix::add(Index row, Index col, Real value)
{
  setValue(row, col, value, ADD_VALUES);
}

void PETScAssemblyMatrix::addAtomic(Index row, Index col, Real value)
{
#pragma omp critical(femx_petsc_matrix_set_value)
  {
    add(row, col, value);
  }
}

void PETScAssemblyMatrix::addBlock(const PetscInt*    dofs,
                           Index              num_dofs,
                           const DenseMatrix& local)
{
  addBlock(dofs, num_dofs, dofs, num_dofs, local);
}

void PETScAssemblyMatrix::addBlock(const PetscInt* dofs, Index num_dofs, const Real* vals)
{
  addBlock(dofs, num_dofs, dofs, num_dofs, vals);
}

void PETScAssemblyMatrix::addBlock(const PetscInt*    rows,
                           Index              num_rows,
                           const PetscInt*    cols,
                           Index              num_cols,
                           const DenseMatrix& local)
{
  if (local.rows() != num_rows || local.cols() != num_cols)
  {
    throw std::runtime_error(
        "PETScAssemblyMatrix local block size does not match dofs");
  }
  addBlock(rows, num_rows, cols, num_cols, local.data());
}

void PETScAssemblyMatrix::addBlock(const PetscInt* rows,
                           Index           num_rows,
                           const PetscInt* cols,
                           Index           num_cols,
                           const Real*     vals)
{
  if (mat_ == nullptr)
  {
    throw std::runtime_error("PETScAssemblyMatrix is not initialized");
  }
  check(MatSetValues(mat_,
                     static_cast<PetscInt>(num_rows),
                     rows,
                     static_cast<PetscInt>(num_cols),
                     cols,
                     vals,
                     ADD_VALUES),
        "MatSetValues");
}

void PETScAssemblyMatrix::finalize()
{
  if (mat_ == nullptr)
  {
    throw std::runtime_error("PETScAssemblyMatrix is not initialized");
  }
  check(MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
}

void PETScAssemblyMatrix::zeroRowsColumns(const Vector<Index>& rows,
                                  Real                 diagonal,
                                  const PETScVector&   vals,
                                  PETScVector&         rhs)
{
  if (vals.size() != rows_ || rhs.size() != rows_)
  {
    throw std::runtime_error(
        "PETScAssemblyMatrix zeroRowsColumns received incompatible vectors");
  }
  if (rows.empty())
  {
    return;
  }

  Vector<PetscInt> petsc_rows;
  petsc_rows.reserve(rows.size());
  for (Index row : rows)
  {
    if (row < 0 || row >= rows_)
    {
      throw std::runtime_error(
          "PETScAssemblyMatrix zeroRowsColumns row is out of range");
    }
    petsc_rows.push_back(static_cast<PetscInt>(row));
  }

  check(MatZeroRowsColumns(mat(),
                           static_cast<PetscInt>(petsc_rows.size()),
                           petsc_rows.data(),
                           static_cast<PetscScalar>(diagonal),
                           vals.vec(),
                           rhs.vec()),
        "MatZeroRowsColumns");
}

void PETScAssemblyMatrix::zeroRows(const Vector<Index>& rows, Real diagonal)
{
  if (rows.empty())
  {
    return;
  }

  Vector<PetscInt> petsc_rows;
  petsc_rows.reserve(rows.size());
  for (Index row : rows)
  {
    if (row < 0 || row >= rows_)
    {
      throw std::runtime_error("PETScAssemblyMatrix zeroRows row is out of range");
    }
    petsc_rows.push_back(static_cast<PetscInt>(row));
  }

  check(MatZeroRows(mat(),
                    static_cast<PetscInt>(petsc_rows.size()),
                    petsc_rows.data(),
                    static_cast<PetscScalar>(diagonal),
                    nullptr,
                    nullptr),
        "MatZeroRows");
}

void PETScAssemblyMatrix::apply(const Vector<Real>& dir, Vector<Real>& out) const
{
  if (dir.size() != numCols())
  {
    throw std::runtime_error(
        "PETScAssemblyMatrix apply received incompatible vector");
  }

  ScopedVec x;
  ScopedVec y;
  createVec(numCols(), x);
  createVec(numRows(), y);
  check(detail::copyToPETSc(dir, x.get()), "copyToPETSc");
  check(MatMult(mat(), x.get(), y.get()), "MatMult");
  check(detail::copyFromPETSc(y.get(), out), "copyFromPETSc");
}

void PETScAssemblyMatrix::applyT(const Vector<Real>& dir, Vector<Real>& out) const
{
  if (dir.size() != numRows())
  {
    throw std::runtime_error(
        "PETScAssemblyMatrix transpose apply received incompatible vector");
  }

  ScopedVec x;
  ScopedVec y;
  createVec(numRows(), x);
  createVec(numCols(), y);
  check(detail::copyToPETSc(dir, x.get()), "copyToPETSc");
  check(MatMultTranspose(mat(), x.get(), y.get()), "MatMultTranspose");
  check(detail::copyFromPETSc(y.get(), out), "copyFromPETSc");
}

PETScAssemblyMatrix::ScopedVec::~ScopedVec()
{
  if (vec_ != nullptr)
  {
    VecDestroy(&vec_);
  }
}

Vec PETScAssemblyMatrix::ScopedVec::get() const
{
  return vec_;
}

Vec* PETScAssemblyMatrix::ScopedVec::put()
{
  return &vec_;
}

void PETScAssemblyMatrix::createVec(Index size, ScopedVec& out) const
{
  PetscMPIInt comm_size = 1;
  checkMPI(MPI_Comm_size(comm_, &comm_size), "MPI_Comm_size");

  const PetscInt global_size = static_cast<PetscInt>(size);
  const PetscInt num_local_dofs        = comm_size == 1 ? global_size : PETSC_DECIDE;

  check(VecCreate(comm_, out.put()), "VecCreate");
  check(VecSetSizes(out.get(), num_local_dofs, global_size), "VecSetSizes");
  check(VecSetFromOptions(out.get()), "VecSetFromOptions");
}

void PETScAssemblyMatrix::setValue(Index row, Index col, Real value, InsertMode mode)
{
  if (mat_ == nullptr)
  {
    throw std::runtime_error("PETScAssemblyMatrix is not initialized");
  }
  check(MatSetValue(mat_,
                    static_cast<PetscInt>(row),
                    static_cast<PetscInt>(col),
                    static_cast<PetscScalar>(value),
                    mode),
        "MatSetValue");
}

void PETScAssemblyMatrix::checkInitialized()
{
  PetscBool initialized = PETSC_FALSE;
  check(PetscInitialized(&initialized), "PetscInitialized");
  if (initialized != PETSC_TRUE)
  {
    throw std::runtime_error("PETScAssemblyMatrix requires initialized PETSc");
  }
}

void PETScAssemblyMatrix::check(PetscErrorCode ierr, const char* operation)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

void PETScAssemblyMatrix::checkMPI(int ierr, const char* operation)
{
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

void PETScAssemblyMatrix::computePreallocation(const CsrPattern& pattern,
                                       PetscInt          begin,
                                       PetscInt          end,
                                       Vector<PetscInt>& d_nnz,
                                       Vector<PetscInt>& o_nnz)
{
  const PetscInt local_rows = end - begin;
  d_nnz.assign(local_rows, 0);
  o_nnz.assign(local_rows, 0);

  const Index* rp = pattern.rowPtrData();
  const Index* ci = pattern.colIndData();
  for (PetscInt row = begin; row < end; ++row)
  {
    PetscInt diagonal = 0;
    PetscInt offdiag  = 0;
    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      const PetscInt col = static_cast<PetscInt>(ci[k]);
      if (col >= begin && col < end)
      {
        ++diagonal;
      }
      else
      {
        ++offdiag;
      }
    }
    d_nnz[row - begin] = diagonal;
    o_nnz[row - begin] = offdiag;
  }
}

} // namespace linalg
} // namespace femx
