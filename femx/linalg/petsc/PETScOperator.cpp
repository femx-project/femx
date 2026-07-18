#include <stdexcept>
#include <string>

#include <femx/linalg/CsrGraph.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>
#include <femx/linalg/petsc/PETScVector.hpp>
#include <femx/linalg/petsc/VectorConversion.hpp>

namespace femx
{
namespace linalg
{

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

Index PETScOperator::numRows() const
{
  return rows_;
}

Index PETScOperator::numCols() const
{
  return cols_;
}

Mat PETScOperator::mat() const
{
  if (mat_ == nullptr)
  {
    throw std::runtime_error("PETScOperator is not initialized");
  }
  return mat_;
}

MPI_Comm PETScOperator::comm() const
{
  return comm_;
}

void PETScOperator::setDefaultNnzPerRow(Index nnz)
{
  if (nnz <= 0)
  {
    throw std::runtime_error("PETScOperator preallocation must be positive");
  }
  default_nnz_per_row_ = nnz;
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

  Index nnz_per_row = cols_ > 0 ? default_nnz_per_row_ : 1;
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

void PETScOperator::resize(const HostCsrGraph& graph,
                           const PETScVector&  lyt)
{
  checkInit();

  if (lyt.size() != graph.rows())
  {
    throw std::runtime_error(
        "PETScOperator row layout does not match CSR graph");
  }

  if (mat_ != nullptr && rows_ == graph.rows() && cols_ == graph.cols())
  {
    setZero();
    return;
  }

  if (mat_ != nullptr)
  {
    check(MatDestroy(&mat_), "MatDestroy");
  }

  comm_ = lyt.comm();
  rows_ = graph.rows();
  cols_ = graph.cols();

  const PetscInt begin = lyt.ownershipBegin();
  const PetscInt end   = lyt.ownershipEnd();

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
  setValue(row, col, val, INSERT_VALUES);
}

void PETScOperator::add(Index row, Index col, Real val)
{
  setValue(row, col, val, ADD_VALUES);
}

void PETScOperator::addAtomic(Index row, Index col, Real val)
{
#pragma omp critical(femx_petsc_matrix_set_value)
  {
    add(row, col, val);
  }
}

void PETScOperator::addBlock(const PetscInt*    dofs,
                             Index              num_dofs,
                             const DenseMatrix& mat_e)
{
  addBlock(dofs, num_dofs, dofs, num_dofs, mat_e);
}

void PETScOperator::addBlock(const PetscInt* dofs, Index num_dofs, const Real* vals)
{
  addBlock(dofs, num_dofs, dofs, num_dofs, vals);
}

void PETScOperator::addBlock(const PetscInt*    rows,
                             Index              num_rows,
                             const PetscInt*    cols,
                             Index              num_cols,
                             const DenseMatrix& mat_e)
{
  if (mat_e.numRows() != num_rows || mat_e.numCols() != num_cols)
  {
    throw std::runtime_error(
        "PETScOperator local block size does not match dofs");
  }
  addBlock(rows, num_rows, cols, num_cols, mat_e.data());
}

void PETScOperator::addBlock(const PetscInt* rows,
                             Index           num_rows,
                             const PetscInt* cols,
                             Index           num_cols,
                             const Real*     vals)
{
  if (mat_ == nullptr)
  {
    throw std::runtime_error("PETScOperator is not initialized");
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

void PETScOperator::finalize()
{
  if (mat_ == nullptr)
  {
    throw std::runtime_error("PETScOperator is not initialized");
  }
  check(MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
}

void PETScOperator::zeroRowsCols(const Array<Index>& rows,
                                 Real                diag,
                                 const PETScVector&  vals,
                                 PETScVector&        rhs)
{
  if (vals.size() != rows_ || rhs.size() != rows_)
  {
    throw std::runtime_error(
        "PETScOperator zeroRowsCols received incompatible vectors");
  }
  if (rows.empty())
  {
    return;
  }

  Array<PetscInt> prows;
  prows.reserve(rows.size());
  for (Index row : rows)
  {
    if (row < 0 || row >= rows_)
    {
      throw std::runtime_error(
          "PETScOperator zeroRowsCols row is out of range");
    }
    prows.push_back(static_cast<PetscInt>(row));
  }

  check(MatZeroRowsColumns(mat(),
                           static_cast<PetscInt>(prows.size()),
                           prows.data(),
                           static_cast<PetscScalar>(diag),
                           vals.vec(),
                           rhs.vec()),
        "MatZeroRowsColumns");
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
    if (row < 0 || row >= rows_)
    {
      throw std::runtime_error("PETScOperator zeroRows row is out of range");
    }
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

void PETScOperator::apply(const HostVector& dir, HostVector& out) const
{
  if (dir.size() != numCols())
  {
    throw std::runtime_error(
        "PETScOperator apply received incompatible vector");
  }

  ScopedVec x;
  ScopedVec y;
  createVec(numCols(), x);
  createVec(numRows(), y);
  check(detail::copyToPETSc(dir, x.get()), "copyToPETSc");
  check(MatMult(mat(), x.get(), y.get()), "MatMult");
  check(detail::copyFromPETSc(y.get(), out), "copyFromPETSc");
}

void PETScOperator::applyT(const HostVector& dir, HostVector& out) const
{
  if (dir.size() != numRows())
  {
    throw std::runtime_error(
        "PETScOperator transpose apply received incompatible vector");
  }

  ScopedVec x;
  ScopedVec y;
  createVec(numRows(), x);
  createVec(numCols(), y);
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

void PETScOperator::setValue(Index row, Index col, Real val, InsertMode mode)
{
  if (mat_ == nullptr)
  {
    throw std::runtime_error("PETScOperator is not initialized");
  }
  check(MatSetValue(mat_,
                    static_cast<PetscInt>(row),
                    static_cast<PetscInt>(col),
                    static_cast<PetscScalar>(val),
                    mode),
        "MatSetValue");
}

void PETScOperator::checkInit()
{
  PetscBool init = PETSC_FALSE;
  check(PetscInitialized(&init), "PetscInitialized");
  if (init != PETSC_TRUE)
  {
    throw std::runtime_error("PETScOperator requires initialized PETSc");
  }
}

void PETScOperator::check(PetscErrorCode ierr, const char* op)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(std::string(op) + " failed");
  }
}

void PETScOperator::checkMPI(int ierr, const char* op)
{
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error(std::string(op) + " failed");
  }
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
