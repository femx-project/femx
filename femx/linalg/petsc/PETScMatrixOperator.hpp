#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/PETScVectorBuilder.hpp>
#include <femx/linalg/petsc/VectorConversion.hpp>

namespace femx
{
namespace linalg
{

/** @brief PETSc-backed matrix operator and assembly target. */
class PETScMatrixOperator final : public MatrixOperator
{
public:
  explicit PETScMatrixOperator(MPI_Comm comm = PETSC_COMM_SELF)
    : comm_(comm)
  {
  }

  PETScMatrixOperator(const PETScMatrixOperator&)            = delete;
  PETScMatrixOperator& operator=(const PETScMatrixOperator&) = delete;

  ~PETScMatrixOperator() override
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
      throw std::runtime_error("PETScMatrixOperator is not initialized");
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
          "PETScMatrixOperator preallocation must be positive");
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

  void resize(const CsrPattern&         pettern,
              const PETScVectorBuilder& lyt)
  {
    checkInitialized();

    if (lyt.size() != pettern.rows())
    {
      throw std::runtime_error(
          "PETScMatrixOperator row layout does not match sparsity pattern");
    }

    if (mat_ != nullptr && rows_ == pettern.rows() && cols_ == pettern.cols())
    {
      setZero();
      return;
    }

    if (mat_ != nullptr)
    {
      check(MatDestroy(&mat_), "MatDestroy");
    }

    comm_ = lyt.comm();
    rows_ = pettern.rows();
    cols_ = pettern.cols();

    PetscInt begin = 0;
    PetscInt end   = 0;
    check(VecGetOwnershipRange(lyt.vec(), &begin, &end),
          "VecGetOwnershipRange");

    Vector<PetscInt> d_nnz;
    Vector<PetscInt> o_nnz;
    computePreallocation(pettern, begin, end, d_nnz, o_nnz);

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

    check(MatSetOption(mat_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE), "MatSetOption");
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
#pragma omp critical(femx_petsc_matrix_set_value)
    {
      add(row, col, value);
    }
  }

  void addBlock(const PetscInt*    dofs,
                Index              nd,
                const DenseMatrix& local)
  {
    addBlock(dofs, nd, dofs, nd, local);
  }

  void addBlock(const PetscInt* dofs,
                Index           nd,
                const Real*     vals)
  {
    addBlock(dofs, nd, dofs, nd, vals);
  }

  void addBlock(const PetscInt*    rows,
                Index              num_rows,
                const PetscInt*    cols,
                Index              num_cols,
                const DenseMatrix& local)
  {
    if (local.rows() != num_rows || local.cols() != num_cols)
    {
      throw std::runtime_error(
          "PETScMatrixOperator local block size does not match dofs");
    }
    addBlock(rows, num_rows, cols, num_cols, local.data());
  }

  void addBlock(const PetscInt* rows,
                Index           num_rows,
                const PetscInt* cols,
                Index           num_cols,
                const Real*     vals)
  {
    if (mat_ == nullptr)
    {
      throw std::runtime_error("PETScMatrixOperator is not initialized");
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

  void finalize() override
  {
    if (mat_ == nullptr)
    {
      throw std::runtime_error("PETScMatrixOperator is not initialized");
    }
    check(MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
    check(MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
  }

  void zeroRowsColumns(const Vector<Index>&      rows,
                       Real                      diagonal,
                       const PETScVectorBuilder& vals,
                       PETScVectorBuilder&       rhs)
  {
    if (vals.size() != rows_ || rhs.size() != rows_)
    {
      throw std::runtime_error(
          "PETScMatrixOperator zeroRowsColumns received incompatible vectors");
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
            "PETScMatrixOperator zeroRowsColumns row is out of range");
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

  void zeroRows(const Vector<Index>& rows, Real diagonal)
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
        throw std::runtime_error(
            "PETScMatrixOperator zeroRows row is out of range");
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

  void apply(const Vector<Real>& dir, Vector<Real>& out) const override
  {
    if (dir.size() != numCols())
    {
      throw std::runtime_error(
          "PETScMatrixOperator apply received incompatible vector");
    }

    ScopedVec x;
    ScopedVec y;
    createVec(numCols(), x);
    createVec(numRows(), y);
    check(detail::copyToPETSc(dir, x.get()), "copyToPETSc");
    check(MatMult(mat(), x.get(), y.get()), "MatMult");
    check(detail::copyFromPETSc(y.get(), out), "copyFromPETSc");
  }

  void applyT(const Vector<Real>& dir, Vector<Real>& out) const override
  {
    if (dir.size() != numRows())
    {
      throw std::runtime_error(
          "PETScMatrixOperator transpose apply received incompatible vector");
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
    const PetscInt nloc        = comm_size == 1 ? global_size : PETSC_DECIDE;

    check(VecCreate(comm_, out.put()), "VecCreate");
    check(VecSetSizes(out.get(), nloc, global_size), "VecSetSizes");
    check(VecSetFromOptions(out.get()), "VecSetFromOptions");
  }

  void setValue(Index      row,
                Index      col,
                Real       value,
                InsertMode mode)
  {
    if (mat_ == nullptr)
    {
      throw std::runtime_error("PETScMatrixOperator is not initialized");
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
      throw std::runtime_error("PETScMatrixOperator requires initialized PETSc");
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

  static void computePreallocation(const CsrPattern& pettern,
                                   PetscInt          begin,
                                   PetscInt          end,
                                   Vector<PetscInt>& d_nnz,
                                   Vector<PetscInt>& o_nnz)
  {
    const PetscInt local_rows = end - begin;
    d_nnz.assign(local_rows, 0);
    o_nnz.assign(local_rows, 0);

    const Index* rp = pettern.rowPtrData();
    const Index* ci = pettern.colIndData();
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

private:
  MPI_Comm comm_{PETSC_COMM_SELF};
  Mat      mat_{nullptr};
  Index    rows_{0};
  Index    cols_{0};
  Index    default_nonzeros_per_row_{32};
};

} // namespace linalg
} // namespace femx
