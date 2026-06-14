#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>
#include <femx/system/petsc/PETScSystemVector.hpp>
#include <femx/system/petsc/VectorConversion.hpp>

namespace femx
{
namespace system
{

/** @brief PETSc-backed implementation of SystemMatrix. */
class PETScSystemMatrix final : public SystemMatrix
{
public:
  explicit PETScSystemMatrix(MPI_Comm comm = PETSC_COMM_SELF)
    : comm_(comm)
  {
  }

  PETScSystemMatrix(const PETScSystemMatrix&)            = delete;
  PETScSystemMatrix& operator=(const PETScSystemMatrix&) = delete;

  ~PETScSystemMatrix() override
  {
    if (mat_ != nullptr)
    {
      MatDestroy(&mat_);
    }
  }

  Index numRows() const override
  {
    return rows_;
  }

  Index numCols() const override
  {
    return cols_;
  }

  Mat mat() const
  {
    if (mat_ == nullptr)
    {
      throw std::runtime_error("PETScSystemMatrix is not initialized");
    }
    return mat_;
  }

  MPI_Comm comm() const
  {
    return comm_;
  }

  void setDefaultNonzerosPerRow(Index count)
  {
    if (count <= 0)
    {
      throw std::runtime_error(
          "PETScSystemMatrix preallocation must be positive");
    }
    default_nonzeros_per_row_ = count;
  }

  void resize(Index rows, Index cols) override
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
    PetscMPIInt    comm_size  = 1;
    checkMPI(MPI_Comm_size(comm_, &comm_size), "MPI_Comm_size");
    const PetscInt local_rows =
        comm_size == 1 ? petsc_rows : PETSC_DECIDE;
    const PetscInt local_cols =
        comm_size == 1 ? petsc_cols : PETSC_DECIDE;
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

  void resize(const CsrPattern&        pattern,
              const PETScSystemVector& layout)
  {
    checkInitialized();

    if (layout.size() != pattern.rows())
    {
      throw std::runtime_error(
          "PETScSystemMatrix row layout does not match sparsity pattern");
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

    comm_ = layout.comm();
    rows_ = pattern.rows();
    cols_ = pattern.cols();

    PetscInt begin = 0;
    PetscInt end   = 0;
    check(VecGetOwnershipRange(layout.vec(), &begin, &end),
          "VecGetOwnershipRange");

    std::vector<PetscInt> d_nnz;
    std::vector<PetscInt> o_nnz;
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

  void setZero() override
  {
    if (mat_ == nullptr)
    {
      return;
    }
    check(MatZeroEntries(mat_), "MatZeroEntries");
  }

  void set(Index row, Index col, Real value) override
  {
    setValue(row, col, value, INSERT_VALUES);
  }

  void add(Index row, Index col, Real value) override
  {
    setValue(row, col, value, ADD_VALUES);
  }

  void addAtomic(Index row, Index col, Real value) override
  {
    add(row, col, value);
  }

  void finalize() override
  {
    if (mat_ == nullptr)
    {
      throw std::runtime_error("PETScSystemMatrix is not initialized");
    }
    check(MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
    check(MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
  }

  void zeroRowsColumns(const std::vector<Index>& rows,
                       Real                      diagonal,
                       const PETScSystemVector&  values,
                       PETScSystemVector&        rhs)
  {
    if (values.size() != rows_ || rhs.size() != rows_)
    {
      throw std::runtime_error(
          "PETScSystemMatrix zeroRowsColumns received incompatible vectors");
    }
    if (rows.empty())
    {
      return;
    }

    std::vector<PetscInt> petsc_rows(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
      if (rows[i] < 0 || rows[i] >= rows_)
      {
        throw std::runtime_error(
            "PETScSystemMatrix zeroRowsColumns row is out of range");
      }
      petsc_rows[i] = static_cast<PetscInt>(rows[i]);
    }

    check(MatZeroRowsColumns(mat(),
                             static_cast<PetscInt>(petsc_rows.size()),
                             petsc_rows.data(),
                             static_cast<PetscScalar>(diagonal),
                             values.vec(),
                             rhs.vec()),
          "MatZeroRowsColumns");
  }

  void apply(const Vector& dir, Vector& out) const override
  {
    if (dir.size() != numCols())
    {
      throw std::runtime_error(
          "PETScSystemMatrix apply received incompatible vector");
    }

    ScopedVec x;
    ScopedVec y;
    createVec(numCols(), x);
    createVec(numRows(), y);
    check(detail::copyToPETSc(dir, x.get()), "copyToPETSc");
    check(MatMult(mat(), x.get(), y.get()), "MatMult");
    check(detail::copyFromPETSc(y.get(), out), "copyFromPETSc");
  }

  void applyT(const Vector& dir, Vector& out) const override
  {
    if (dir.size() != numRows())
    {
      throw std::runtime_error(
          "PETScSystemMatrix transpose apply received incompatible vector");
    }

    ScopedVec x;
    ScopedVec y;
    createVec(numRows(), x);
    createVec(numCols(), y);
    check(detail::copyToPETSc(dir, x.get()), "copyToPETSc");
    check(MatMultTranspose(mat(), x.get(), y.get()), "MatMultTranspose");
    check(detail::copyFromPETSc(y.get(), out), "copyFromPETSc");
  }

private:
  class ScopedVec
  {
  public:
    ~ScopedVec()
    {
      if (vec_ != nullptr)
      {
        VecDestroy(&vec_);
      }
    }

    Vec get() const
    {
      return vec_;
    }

    Vec* put()
    {
      return &vec_;
    }

  private:
    Vec vec_{nullptr};
  };

  void createVec(Index size, ScopedVec& out) const
  {
    PetscMPIInt comm_size = 1;
    checkMPI(MPI_Comm_size(comm_, &comm_size), "MPI_Comm_size");
    const PetscInt global_size = static_cast<PetscInt>(size);
    const PetscInt local_size =
        comm_size == 1 ? global_size : PETSC_DECIDE;
    check(VecCreate(comm_, out.put()), "VecCreate");
    check(VecSetSizes(out.get(), local_size, global_size), "VecSetSizes");
    check(VecSetFromOptions(out.get()), "VecSetFromOptions");
  }

  void setValue(Index      row,
                Index      col,
                Real       value,
                InsertMode mode)
  {
    if (mat_ == nullptr)
    {
      throw std::runtime_error("PETScSystemMatrix is not initialized");
    }
    check(MatSetValue(mat_,
                      static_cast<PetscInt>(row),
                      static_cast<PetscInt>(col),
                      static_cast<PetscScalar>(value),
                      mode),
          "MatSetValue");
  }

  static void checkInitialized()
  {
    PetscBool initialized = PETSC_FALSE;
    check(PetscInitialized(&initialized), "PetscInitialized");
    if (initialized != PETSC_TRUE)
    {
      throw std::runtime_error("PETScSystemMatrix requires initialized PETSc");
    }
  }

  static void check(PetscErrorCode ierr, const char* operation)
  {
    if (ierr != PETSC_SUCCESS)
    {
      throw std::runtime_error(std::string(operation) + " failed");
    }
  }

  static void checkMPI(int ierr, const char* operation)
  {
    if (ierr != MPI_SUCCESS)
    {
      throw std::runtime_error(std::string(operation) + " failed");
    }
  }

  static void computePreallocation(const CsrPattern&      pattern,
                                   PetscInt               begin,
                                   PetscInt               end,
                                   std::vector<PetscInt>& d_nnz,
                                   std::vector<PetscInt>& o_nnz)
  {
    const PetscInt local_rows = end - begin;
    d_nnz.assign(static_cast<std::size_t>(local_rows), 0);
    o_nnz.assign(static_cast<std::size_t>(local_rows), 0);

    const Index* row_ptr = pattern.rowPtrData();
    const Index* col_ind = pattern.colIndData();
    for (PetscInt row = begin; row < end; ++row)
    {
      PetscInt diagonal = 0;
      PetscInt offdiag  = 0;
      for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
      {
        const PetscInt col = static_cast<PetscInt>(col_ind[k]);
        if (col >= begin && col < end)
        {
          ++diagonal;
        }
        else
        {
          ++offdiag;
        }
      }
      d_nnz[static_cast<std::size_t>(row - begin)] = diagonal;
      o_nnz[static_cast<std::size_t>(row - begin)] = offdiag;
    }
  }

private:
  MPI_Comm comm_{PETSC_COMM_SELF};
  Mat      mat_{nullptr};
  Index    rows_{0};
  Index    cols_{0};
  Index    default_nonzeros_per_row_{32};
};

} // namespace system
} // namespace femx
