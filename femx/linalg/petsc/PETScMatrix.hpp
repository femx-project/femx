#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <femx/common/Types.hpp>

namespace femx
{
class DenseMatrix;

namespace linalg
{

class PETScSystemMatrix;

/**
 * @brief Own and assemble a PETSc-backed matrix.
 *
 * PETScMatrix owns a PETSc Mat and provides assembly, preallocation,
 * row elimination, and matrix-vector application helpers.
 */
class PETScMatrix final
{
public:
  /**
   * @brief Construct an empty operator on a communicator.
   *
   * @param[in] comm - PETSc communicator.
   */
  explicit PETScMatrix(MPI_Comm comm = PETSC_COMM_SELF);

  PETScMatrix(const PETScMatrix&) = delete;

  PETScMatrix& operator=(const PETScMatrix&) = delete;

  ~PETScMatrix();

  /** @brief Return the global number of rows. */
  Index rows() const;

  /** @brief Return the global number of columns. */
  Index cols() const;

  /**
   * @brief Return the initialized PETSc matrix handle.
   *
   * @return Borrowed PETSc matrix handle.
   * @throws std::runtime_error - If the operator is not initialized.
   */
  Mat mat() const;

  /** @brief Return the PETSc communicator used by the matrix. */
  MPI_Comm comm() const;

  /**
   * @brief Allocate an AIJ matrix with fallback preallocation.
   *
   * @param[in] rows - Global number of rows.
   * @param[in] cols - Global number of columns.
   * @throws std::runtime_error - If PETSc is not initialized or PETSc or MPI
   * reports an error.
   */
  void resize(Index rows, Index cols);

  /**
   * @brief Allocate from an exact Host CSR pattern.
   *
   * @param[in] pattern - CSR pattern used for preallocation.
   * @throws std::runtime_error - If PETSc is not initialized or PETSc or MPI
   * reports an error.
   */
  void resize(const HostCsrPattern& pattern);

  /**
   * @brief Zero all numeric entries while retaining sparsity.
   *
   * @throws std::runtime_error - If PETSc reports an error.
   */
  void setZero();

  /**
   * @brief Replace one global entry before finalization.
   *
   * @param[in] row - Global row index.
   * @param[in] col - Global column index.
   * @param[in] val - Replacement value.
   * @throws std::runtime_error - If the operator is uninitialized or PETSc
   * reports an error.
   */
  void set(Index row, Index col, Real val);

  /**
   * @brief Add a dense block using global index arrays.
   *
   * @param[in] rows - Global row indices.
   * @param[in] cols - Global column indices.
   * @param[in] mat_e - Dense values matching the index arrays.
   * @throws std::runtime_error - If dimensions are inconsistent, the operator
   * is uninitialized, or PETSc reports an error.
   */
  void addBlock(HostVectorView<const Index> rows,
                HostVectorView<const Index> columns,
                HostMatrixView<const Real>  values);

  /**
   * @brief Complete PETSc matrix assembly.
   *
   * @throws std::runtime_error - If the operator is uninitialized or PETSc
   * reports an error.
   */
  void finalize();

  /**
   * @brief Replace selected rows with diagonal constraints.
   *
   * @param[in] rows - Global rows to replace.
   * @param[in] diag - Replacement diagonal value.
   * @throws std::runtime_error - If a row is invalid or PETSc reports an error.
   */
  void replaceRows(HostVectorView<const Index> rows, Real diagonal);

  /**
   * @brief Eliminate constrained rows and columns with RHS correction.
   *
   * @param[in] rows - Global constrained rows.
   * @param[in] values - Prescribed values in row order.
   * @param[in,out] rhs - Replicated Host right-hand side.
   */
  void eliminateColumns(HostVectorView<const Index> rows,
                        HostVectorView<const Real>  values,
                        HostVectorView<Real>        rhs);

  /**
   * @brief Apply the matrix to a replicated Host vector.
   *
   * @param[in] dir - Input vector.
   * @param[out] out - Replicated result vector.
   * @throws std::runtime_error - If dimensions are inconsistent, the operator
   * is uninitialized, or PETSc reports an error.
   */
  void apply(HostVectorView<const Real> dir, HostVector<Real>& out) const;

  /**
   * @brief Apply the transpose to a replicated Host vector.
   *
   * @param[in] dir - Input vector.
   * @param[out] out - Replicated result vector.
   * @throws std::runtime_error - If dimensions are inconsistent, the operator
   * is uninitialized, or PETSc reports an error.
   */
  void applyT(HostVectorView<const Real> dir, HostVector<Real>& out) const;

private:
  class ScopedVec
  {
  public:
    ~ScopedVec();

    Vec get() const;

    Vec* put();

  private:
    Vec vec_{nullptr}; ///< Owned PETSc vector handle.
  };

  void createVec(Index size, ScopedVec& out) const;

  void addBlock(const PetscInt* rows,
                Index           num_rows,
                const PetscInt* cols,
                Index           num_cols,
                const Real*     vals);

  void zeroRows(HostVectorView<const Index> rows, Real diagonal);

  static void computePrealloc(const HostCsrPattern& pattern,
                              PetscInt              begin,
                              PetscInt              end,
                              HostVector<PetscInt>& diag_nnz,
                              HostVector<PetscInt>& offdiag_nnz);

private:
  MPI_Comm comm_{PETSC_COMM_SELF}; ///< Communicator used by the matrix.
  Mat      mat_{nullptr};          ///< Owned PETSc matrix handle.
  Index    rows_{0};               ///< Global row count.
  Index    cols_{0};               ///< Global column count.
};

} // namespace linalg
} // namespace femx
