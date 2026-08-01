#include <stdexcept>
#include <string>

#include <femx/common/Checks.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/petsc/PETScMatrix.hpp>
#include <femx/linalg/petsc/PETScPartition.hpp>
#include <femx/linalg/petsc/PETScUtilities.hpp>

namespace femx
{
namespace linalg
{

void detail::check(PetscErrorCode ierr, const char* op)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(std::string(op) + " failed");
  }
}

void detail::checkMPI(int ierr, const char* op)
{
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error(std::string(op) + " failed");
  }
}

void detail::checkInit()
{
  PetscBool init = PETSC_FALSE;
  check(PetscInitialized(&init), "PetscInitialized");
  require(init == PETSC_TRUE, "PETSc must be initialized");
}

PetscErrorCode detail::copyFromPETSc(
    Vec                                          src,
    HostVector<Real>&                            dst,
    const std::shared_ptr<const PETScPartition>& partition)
{
  PetscInt size = 0;
  PetscCall(VecGetSize(src, &size));
  if (partition && partition->size() != size)
  {
    return PETSC_ERR_ARG_SIZ;
  }
  dst.resize(static_cast<Index>(size));

  VecScatter scatter = nullptr;
  Vec        all     = nullptr;
  PetscCall(VecScatterCreateToAll(src, &scatter, &all));
  PetscCall(VecScatterBegin(
      scatter, src, all, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(
      scatter, src, all, INSERT_VALUES, SCATTER_FORWARD));

  const PetscScalar* vals = nullptr;
  PetscCall(VecGetArrayRead(all, &vals));
  for (PetscInt i = 0; i < size; ++i)
  {
    const Index app_index =
        partition ? partition->applicationIndex(i)
                  : static_cast<Index>(i);
    dst[app_index] = PetscRealPart(vals[i]);
  }
  PetscCall(VecRestoreArrayRead(all, &vals));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&all));
  return PETSC_SUCCESS;
}

PetscErrorCode detail::copyToPETSc(
    HostVectorView<const Real>                   src,
    Vec                                          dst,
    const std::shared_ptr<const PETScPartition>& partition)
{
  PetscInt size = 0;
  PetscCall(VecGetSize(dst, &size));
  if (src.size() != static_cast<Index>(size)
      || (partition && partition->size() != size))
  {
    return PETSC_ERR_ARG_SIZ;
  }

  PetscInt begin = 0;
  PetscInt end   = 0;
  PetscCall(VecGetOwnershipRange(dst, &begin, &end));
  PetscScalar* vals = nullptr;
  PetscCall(VecGetArray(dst, &vals));
  for (PetscInt i = begin; i < end; ++i)
  {
    const Index app_index =
        partition ? partition->applicationIndex(i)
                  : static_cast<Index>(i);
    vals[i - begin] =
        static_cast<PetscScalar>(src[app_index]);
  }
  PetscCall(VecRestoreArray(dst, &vals));
  return PETSC_SUCCESS;
}

using detail::check;
using detail::checkInit;
using detail::checkMPI;

PETScMatrix::PETScMatrix(MPI_Comm comm)
  : comm_(comm)
{
}

PETScMatrix::~PETScMatrix()
{
  if (mat_ != nullptr)
  {
    MatDestroy(&mat_);
  }
}

Index PETScMatrix::rows() const
{
  return rows_;
}

Index PETScMatrix::cols() const
{
  return cols_;
}

Mat PETScMatrix::mat() const
{
  require(mat_ != nullptr, "PETScMatrix is not initialized");
  return mat_;
}

MPI_Comm PETScMatrix::comm() const
{
  return comm_;
}

std::shared_ptr<const PETScPartition>
PETScMatrix::partition() const noexcept
{
  return partition_;
}

void PETScMatrix::resize(Index rows, Index cols)
{
  checkInit();

  if (mat_ != nullptr && layout_id_ == 0 && !partition_
      && rows_ == rows && cols_ == cols)
  {
    setZero();
    return;
  }

  if (mat_ != nullptr)
  {
    check(MatDestroy(&mat_), "MatDestroy");
  }

  rows_      = rows;
  cols_      = cols;
  layout_id_ = 0;
  partition_.reset();

  const PetscInt nrow = static_cast<PetscInt>(rows_);
  const PetscInt ncol = static_cast<PetscInt>(cols_);

  PetscMPIInt comm_size = 1;
  checkMPI(MPI_Comm_size(comm_, &comm_size), "MPI_Comm_size");

  const PetscInt nrow_local = comm_size == 1 ? nrow : PETSC_DECIDE;
  const PetscInt ncol_local = comm_size == 1 ? ncol : PETSC_DECIDE;

  constexpr Index kDefaultNnzPerRow = 32;
  Index           nnz_per_row       = cols_ > 0 ? kDefaultNnzPerRow : 1;
  if (cols_ > 0 && cols_ < nnz_per_row)
  {
    nnz_per_row = cols_;
  }

  check(MatCreateAIJ(comm_,
                     nrow_local,
                     ncol_local,
                     nrow,
                     ncol,
                     static_cast<PetscInt>(nnz_per_row),
                     nullptr,
                     static_cast<PetscInt>(nnz_per_row),
                     nullptr,
                     &mat_),
        "MatCreateAIJ");
  check(MatSetOption(mat_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE),
        "MatSetOption");
  check(MatSetOption(mat_, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE),
        "MatSetOption");
}

void PETScMatrix::resize(const HostCsrPattern& pattern)
{
  checkInit();

  if (mat_ != nullptr && layout_id_ == pattern.layoutId())
  {
    setZero();
    return;
  }

  if (mat_ != nullptr)
  {
    check(MatDestroy(&mat_), "MatDestroy");
  }

  rows_      = pattern.rows();
  cols_      = pattern.cols();
  layout_id_ = pattern.layoutId();

  PetscInt begin      = 0;
  PetscInt local_rows = PETSC_DECIDE;
  PetscInt local_cols = PETSC_DECIDE;

  HostVector<PetscInt> diag_nnz;
  HostVector<PetscInt> offdiag_nnz;
  if (rows_ == cols_)
  {
    partition_ = PETScPartition::create(comm_, pattern);
    begin      = partition_->begin();
    local_rows = partition_->localSize();
    local_cols = partition_->localSize();
    computePrealloc(pattern, *partition_, diag_nnz, offdiag_nnz);
  }
  else
  {
    partition_.reset();
    PetscInt global_rows = static_cast<PetscInt>(rows_);
    check(PetscSplitOwnership(comm_, &local_rows, &global_rows),
          "PetscSplitOwnership");

    checkMPI(MPI_Exscan(&local_rows,
                        &begin,
                        1,
                        MPIU_INT,
                        MPI_SUM,
                        comm_),
             "MPI_Exscan");
    PetscMPIInt rank = 0;
    checkMPI(MPI_Comm_rank(comm_, &rank), "MPI_Comm_rank");
    if (rank == 0)
    {
      begin = 0;
    }
    computePrealloc(pattern,
                    begin,
                    begin + local_rows,
                    diag_nnz,
                    offdiag_nnz);
  }

  check(MatCreateAIJ(comm_,
                     local_rows,
                     local_cols,
                     static_cast<PetscInt>(rows_),
                     static_cast<PetscInt>(cols_),
                     0,
                     diag_nnz.empty() ? nullptr : diag_nnz.data(),
                     0,
                     offdiag_nnz.empty() ? nullptr : offdiag_nnz.data(),
                     &mat_),
        "MatCreateAIJ");

  check(MatSetOption(mat_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE),
        "MatSetOption");
  check(MatSetOption(mat_, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE),
        "MatSetOption");
  check(MatSetUp(mat_), "MatSetUp");
}

void PETScMatrix::setZero()
{
  if (mat_ == nullptr)
  {
    return;
  }
  check(MatZeroEntries(mat_), "MatZeroEntries");
}

void PETScMatrix::set(Index row, Index col, Real val)
{
  require(mat_ != nullptr, "PETScMatrix is not initialized");
  check(MatSetValue(mat_,
                    mappedIndex(row),
                    mappedIndex(col),
                    static_cast<PetscScalar>(val),
                    INSERT_VALUES),
        "MatSetValue");
}

void PETScMatrix::addBlock(HostVectorView<const Index> rows,
                           HostVectorView<const Index> columns,
                           HostMatrixView<const Real>  values)
{
  require(values.rows() == rows.size()
              && values.cols() == columns.size(),
          "PETScMatrix local block size does not match dofs");
  static thread_local HostVector<PetscInt> petsc_rows;
  static thread_local HostVector<PetscInt> petsc_cols;
  petsc_rows.resize(rows.size());
  petsc_cols.resize(columns.size());
  for (Index i = 0; i < rows.size(); ++i)
  {
    petsc_rows[i] = mappedIndex(rows[i]);
  }
  for (Index i = 0; i < columns.size(); ++i)
  {
    petsc_cols[i] = mappedIndex(columns[i]);
  }
  addBlock(petsc_rows.data(),
           petsc_rows.size(),
           petsc_cols.data(),
           petsc_cols.size(),
           values.data());
}

void PETScMatrix::addBlock(const PetscInt* rows,
                           Index           num_rows,
                           const PetscInt* cols,
                           Index           num_cols,
                           const Real*     vals)
{
  require(mat_ != nullptr, "PETScMatrix is not initialized");
  check(MatSetValues(mat_,
                     static_cast<PetscInt>(num_rows),
                     rows,
                     static_cast<PetscInt>(num_cols),
                     cols,
                     vals,
                     ADD_VALUES),
        "MatSetValues");
}

void PETScMatrix::finalize()
{
  require(mat_ != nullptr, "PETScMatrix is not initialized");
  check(MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
}

void PETScMatrix::replaceRows(HostVectorView<const Index> rows,
                              Real                        diag)
{
  finalize();
  zeroRows(rows, diag);
}

void PETScMatrix::zeroRows(HostVectorView<const Index> rows,
                           Real                        diag)
{
  if (rows.empty())
  {
    return;
  }

  HostVector<PetscInt> prows;
  prows.reserve(rows.size());
  for (Index row : rows)
  {
    require(row >= 0 && row < rows_,
            "PETScMatrix zeroRows row is out of range");
    prows.push_back(mappedIndex(row));
  }

  check(MatZeroRows(mat(),
                    static_cast<PetscInt>(prows.size()),
                    prows.data(),
                    static_cast<PetscScalar>(diag),
                    nullptr,
                    nullptr),
        "MatZeroRows");
}

void PETScMatrix::eliminateColumns(
    HostVectorView<const Index> rows,
    HostVectorView<const Real>  values,
    HostVectorView<Real>        rhs)
{
  require(values.size() == rows.size() && rhs.size() == rows_,
          "PETSc matrix constraint vectors have incompatible dimensions");
  if (rows.empty())
  {
    return;
  }

  finalize();
  HostVector<PetscInt> petsc_rows(rows.size());
  HostVector<Real>     prescribed(cols_, 0.0);
  for (Index i = 0; i < rows.size(); ++i)
  {
    require(rows[i] >= 0 && rows[i] < rows_,
            "PETSc matrix constrained row is out of range");
    petsc_rows[i]       = mappedIndex(rows[i]);
    prescribed[rows[i]] = values[i];
  }

  ScopedVec prescribed_vector;
  ScopedVec rhs_vector;
  createVec(cols_, prescribed_vector);
  createVec(rows_, rhs_vector);
  check(detail::copyToPETSc(
            prescribed.view(), prescribed_vector.get(), partition_),
        "copyToPETSc");
  check(detail::copyToPETSc(rhs, rhs_vector.get(), partition_),
        "copyToPETSc");
  check(MatZeroRowsColumns(mat(),
                           static_cast<PetscInt>(petsc_rows.size()),
                           petsc_rows.data(),
                           1.0,
                           prescribed_vector.get(),
                           rhs_vector.get()),
        "MatZeroRowsColumns");

  HostVector<Real> corrected;
  check(detail::copyFromPETSc(
            rhs_vector.get(), corrected, partition_),
        "copyFromPETSc");
  rhs = corrected;
}

void PETScMatrix::matvec(HostVectorView<const Real> dir,
                         HostVector<Real>&          out) const
{
  require(dir.size() == cols(),
          "PETScMatrix apply received incompatible vector");

  ScopedVec x;
  ScopedVec y;
  createVec(cols(), x);
  createVec(rows(), y);
  check(detail::copyToPETSc(dir, x.get(), partition_), "copyToPETSc");
  check(MatMult(mat(), x.get(), y.get()), "MatMult");
  check(detail::copyFromPETSc(y.get(), out, partition_), "copyFromPETSc");
}

void PETScMatrix::matvecT(HostVectorView<const Real> dir,
                          HostVector<Real>&          out) const
{
  require(dir.size() == rows(),
          "PETScMatrix transpose apply received incompatible vector");

  ScopedVec x;
  ScopedVec y;
  createVec(rows(), x);
  createVec(cols(), y);
  check(detail::copyToPETSc(dir, x.get(), partition_), "copyToPETSc");
  check(MatMultTranspose(mat(), x.get(), y.get()), "MatMultTranspose");
  check(detail::copyFromPETSc(y.get(), out, partition_), "copyFromPETSc");
}

PETScMatrix::ScopedVec::~ScopedVec()
{
  if (vec_ != nullptr)
  {
    VecDestroy(&vec_);
  }
}

Vec PETScMatrix::ScopedVec::get() const
{
  return vec_;
}

Vec* PETScMatrix::ScopedVec::put()
{
  return &vec_;
}

void PETScMatrix::createVec(Index size, ScopedVec& out) const
{
  PetscMPIInt comm_size = 1;
  checkMPI(MPI_Comm_size(comm_, &comm_size), "MPI_Comm_size");

  const PetscInt n = static_cast<PetscInt>(size);
  const PetscInt n_local =
      partition_ && partition_->size() == size
          ? partition_->localSize()
          : (comm_size == 1 ? n : PETSC_DECIDE);

  check(VecCreate(comm_, out.put()), "VecCreate");
  check(VecSetSizes(out.get(), n_local, n), "VecSetSizes");
  check(VecSetFromOptions(out.get()), "VecSetFromOptions");
}

void PETScMatrix::computePrealloc(const HostCsrPattern& pattern,
                                  PetscInt              begin,
                                  PetscInt              end,
                                  HostVector<PetscInt>& diag_nnz,
                                  HostVector<PetscInt>& offdiag_nnz)
{
  const PetscInt nrow = end - begin;
  diag_nnz.assign(nrow, 0);
  offdiag_nnz.assign(nrow, 0);

  const Index* rp = pattern.rowPtrData();
  const Index* ci = pattern.colIndData();
  for (PetscInt row = begin; row < end; ++row)
  {
    PetscInt diag = 0;
    PetscInt off  = 0;
    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      const PetscInt col = static_cast<PetscInt>(ci[k]);
      if (col >= begin && col < end)
      {
        ++diag;
      }
      else
      {
        ++off;
      }
    }
    diag_nnz[row - begin]    = diag;
    offdiag_nnz[row - begin] = off;
  }
}

void PETScMatrix::computePrealloc(
    const HostCsrPattern& pattern,
    const PETScPartition& partition,
    HostVector<PetscInt>& diag_nnz,
    HostVector<PetscInt>& offdiag_nnz)
{
  diag_nnz.assign(partition.localSize(), 0);
  offdiag_nnz.assign(partition.localSize(), 0);

  const Index* rp = pattern.rowPtrData();
  const Index* ci = pattern.colIndData();
  for (Index app_row = 0; app_row < pattern.rows(); ++app_row)
  {
    const PetscInt row = partition.petscIndex(app_row);
    if (row < partition.begin() || row >= partition.end())
    {
      continue;
    }

    PetscInt diag = 0;
    PetscInt off  = 0;
    for (Index k = rp[app_row]; k < rp[app_row + 1]; ++k)
    {
      const PetscInt col = partition.petscIndex(ci[k]);
      if (col >= partition.begin() && col < partition.end())
      {
        ++diag;
      }
      else
      {
        ++off;
      }
    }
    diag_nnz[row - partition.begin()]    = diag;
    offdiag_nnz[row - partition.begin()] = off;
  }
}

PetscInt PETScMatrix::mappedIndex(Index index) const
{
  return partition_ ? partition_->petscIndex(index)
                    : static_cast<PetscInt>(index);
}

} // namespace linalg
} // namespace femx
