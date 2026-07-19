#include <stdexcept>
#include <string>

#include <femx/common/Checks.hpp>
#include <femx/linalg/CsrGraph.hpp>
#include <femx/linalg/Dense.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/PETScBackend.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>

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

PetscErrorCode detail::copyFromPETSc(Vec src, HostVector& dst)
{
  PetscInt size = 0;
  PetscCall(VecGetSize(src, &size));
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
    dst[static_cast<Index>(i)] = PetscRealPart(vals[i]);
  }
  PetscCall(VecRestoreArrayRead(all, &vals));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&all));
  return PETSC_SUCCESS;
}

PetscErrorCode detail::copyToPETSc(HostConstVectorView src, Vec dst)
{
  PetscInt size = 0;
  PetscCall(VecGetSize(dst, &size));
  if (src.size() != static_cast<Index>(size))
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
    vals[i - begin] =
        static_cast<PetscScalar>(src[static_cast<Index>(i)]);
  }
  PetscCall(VecRestoreArray(dst, &vals));
  return PETSC_SUCCESS;
}

using detail::check;
using detail::checkInit;
using detail::checkMPI;

PETScOperator::PETScOperator(MPI_Comm comm)
  : comm_(comm)
{
}

PETScOperator::~PETScOperator()
{
  if (mat_ != nullptr)
  {
    MatDestroy(&mat_);
  }
}

Index PETScOperator::rows() const
{
  return rows_;
}

Index PETScOperator::cols() const
{
  return cols_;
}

Mat PETScOperator::mat() const
{
  require(mat_ != nullptr, "PETScOperator is not initialized");
  return mat_;
}

MPI_Comm PETScOperator::comm() const
{
  return comm_;
}

void PETScOperator::resize(Index rows, Index cols)
{
  checkInit();

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
}

void PETScOperator::resize(const HostCsrGraph& graph)
{
  checkInit();

  if (mat_ != nullptr && rows_ == graph.rows() && cols_ == graph.cols())
  {
    setZero();
    return;
  }

  if (mat_ != nullptr)
  {
    check(MatDestroy(&mat_), "MatDestroy");
  }

  rows_ = graph.rows();
  cols_ = graph.cols();

  PetscInt local_rows  = PETSC_DECIDE;
  PetscInt global_rows = static_cast<PetscInt>(rows_);
  check(PetscSplitOwnership(comm_, &local_rows, &global_rows),
        "PetscSplitOwnership");

  PetscInt begin = 0;
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
  const PetscInt end = begin + local_rows;

  Array<PetscInt> d_nnz;
  Array<PetscInt> o_nnz;
  computePrealloc(graph, begin, end, d_nnz, o_nnz);

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

void PETScOperator::setZero()
{
  if (mat_ == nullptr)
  {
    return;
  }
  check(MatZeroEntries(mat_), "MatZeroEntries");
}

void PETScOperator::set(Index row, Index col, Real val)
{
  require(mat_ != nullptr, "PETScOperator is not initialized");
  check(MatSetValue(mat_,
                    static_cast<PetscInt>(row),
                    static_cast<PetscInt>(col),
                    static_cast<PetscScalar>(val),
                    INSERT_VALUES),
        "MatSetValue");
}

void PETScOperator::addBlock(const Array<Index>& rows,
                             const Array<Index>& cols,
                             const DenseMatrix&  mat_e)
{
  require(mat_e.rows() == rows.size() && mat_e.cols() == cols.size(),
          "PETScOperator local block size does not match dofs");
  static thread_local Array<PetscInt> petsc_rows;
  static thread_local Array<PetscInt> petsc_cols;
  petsc_rows.resize(rows.size());
  petsc_cols.resize(cols.size());
  for (Index i = 0; i < rows.size(); ++i)
  {
    petsc_rows[i] = static_cast<PetscInt>(rows[i]);
  }
  for (Index i = 0; i < cols.size(); ++i)
  {
    petsc_cols[i] = static_cast<PetscInt>(cols[i]);
  }
  addBlock(petsc_rows.data(),
           petsc_rows.size(),
           petsc_cols.data(),
           petsc_cols.size(),
           mat_e.data());
}

void PETScOperator::addBlock(const PetscInt* rows,
                             Index           num_rows,
                             const PetscInt* cols,
                             Index           num_cols,
                             const Real*     vals)
{
  require(mat_ != nullptr, "PETScOperator is not initialized");
  check(MatSetValues(mat_,
                     static_cast<PetscInt>(num_rows),
                     rows,
                     static_cast<PetscInt>(num_cols),
                     cols,
                     vals,
                     ADD_VALUES),
        "MatSetValues");
}

void PETScOperator::finalize()
{
  require(mat_ != nullptr, "PETScOperator is not initialized");
  check(MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
}

void PETScOperator::replaceRows(const Array<Index>& rows, Real diag)
{
  finalize();
  zeroRows(rows, diag);
}

void PETScOperator::zeroRows(const Array<Index>& rows, Real diag)
{
  if (rows.empty())
  {
    return;
  }

  Array<PetscInt> prows;
  prows.reserve(rows.size());
  for (Index row : rows)
  {
    require(row >= 0 && row < rows_,
            "PETScOperator zeroRows row is out of range");
    prows.push_back(static_cast<PetscInt>(row));
  }

  check(MatZeroRows(mat(),
                    static_cast<PetscInt>(prows.size()),
                    prows.data(),
                    static_cast<PetscScalar>(diag),
                    nullptr,
                    nullptr),
        "MatZeroRows");
}

void PETScOperator::apply(HostConstVectorView dir, HostVector& out) const
{
  require(dir.size() == cols(),
          "PETScOperator apply received incompatible vector");

  ScopedVec x;
  ScopedVec y;
  createVec(cols(), x);
  createVec(rows(), y);
  check(detail::copyToPETSc(dir, x.get()), "copyToPETSc");
  check(MatMult(mat(), x.get(), y.get()), "MatMult");
  check(detail::copyFromPETSc(y.get(), out), "copyFromPETSc");
}

void PETScOperator::applyT(HostConstVectorView dir, HostVector& out) const
{
  require(dir.size() == rows(),
          "PETScOperator transpose apply received incompatible vector");

  ScopedVec x;
  ScopedVec y;
  createVec(rows(), x);
  createVec(cols(), y);
  check(detail::copyToPETSc(dir, x.get()), "copyToPETSc");
  check(MatMultTranspose(mat(), x.get(), y.get()), "MatMultTranspose");
  check(detail::copyFromPETSc(y.get(), out), "copyFromPETSc");
}

PETScOperator::ScopedVec::~ScopedVec()
{
  if (vec_ != nullptr)
  {
    VecDestroy(&vec_);
  }
}

Vec PETScOperator::ScopedVec::get() const
{
  return vec_;
}

Vec* PETScOperator::ScopedVec::put()
{
  return &vec_;
}

void PETScOperator::createVec(Index size, ScopedVec& out) const
{
  PetscMPIInt comm_size = 1;
  checkMPI(MPI_Comm_size(comm_, &comm_size), "MPI_Comm_size");

  const PetscInt n       = static_cast<PetscInt>(size);
  const PetscInt n_local = comm_size == 1 ? n : PETSC_DECIDE;

  check(VecCreate(comm_, out.put()), "VecCreate");
  check(VecSetSizes(out.get(), n_local, n), "VecSetSizes");
  check(VecSetFromOptions(out.get()), "VecSetFromOptions");
}

void PETScOperator::computePrealloc(const HostCsrGraph& graph,
                                    PetscInt            begin,
                                    PetscInt            end,
                                    Array<PetscInt>&    d_nnz,
                                    Array<PetscInt>&    o_nnz)
{
  const PetscInt nrow = end - begin;
  d_nnz.assign(nrow, 0);
  o_nnz.assign(nrow, 0);

  const Index* rp = graph.rowPtrData();
  const Index* ci = graph.colIndData();
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
    d_nnz[row - begin] = diag;
    o_nnz[row - begin] = off;
  }
}

} // namespace linalg
} // namespace femx
