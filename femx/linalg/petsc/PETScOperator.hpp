#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <femx/common/Types.hpp>

namespace femx
{
class DenseMatrix;

namespace linalg
{

/**
 * @brief Own and assemble a PETSc-backed matrix.
 *
 * PETScOperator owns a PETSc Mat and provides assembly, preallocation,
 * row elimination, and matrix-vector application helpers.
 */
class PETScOperator final
{
public:
  /**
   * @brief Construct an empty operator on a communicator.
   *
   * @param[in] comm - PETSc communicator.
   */
  explicit PETScOperator(MPI_Comm comm = PETSC_COMM_SELF);

  PETScOperator(const PETScOperator&) = delete;

  PETScOperator& operator=(const PETScOperator&) = delete;

  ~PETScOperator();

  /**
   * @brief Return the global number of rows.
   *
   * @return Number of rows.
   */
  Index rows() const;

  /**
   * @brief Return the global number of columns.
   *
   * @return Number of columns.
   */
  Index cols() const;

  /**
   * @brief Return the initialized PETSc matrix handle.
   *
   * @return Borrowed PETSc matrix handle.
   * @throws std::runtime_error - If the operator is not initialized.
   */
  Mat mat() const;

  /**
   * @brief Return the PETSc communicator.
   *
   * @return Communicator used by the matrix.
   */
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
  void addBlock(const Array<Index>& rows,
                const Array<Index>& cols,
                const DenseMatrix&  mat_e);

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
  void replaceRows(const Array<Index>& rows, Real diag);

  /**
   * @brief Apply the matrix to a replicated Host vector.
   *
   * @param[in] dir - Input vector.
   * @param[out] out - Replicated result vector.
   * @throws std::runtime_error - If dimensions are inconsistent, the operator
   * is uninitialized, or PETSc reports an error.
   */
  void apply(HostConstVectorView dir, HostVector& out) const;

  /**
   * @brief Apply the transpose to a replicated Host vector.
   *
   * @param[in] dir - Input vector.
   * @param[out] out - Replicated result vector.
   * @throws std::runtime_error - If dimensions are inconsistent, the operator
   * is uninitialized, or PETSc reports an error.
   */
  void applyT(HostConstVectorView dir, HostVector& out) const;

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

  void zeroRows(const Array<Index>& rows, Real diag);

  static void computePrealloc(const HostCsrPattern& pattern,
                              PetscInt              begin,
                              PetscInt              end,
                              Array<PetscInt>&      d_nnz,
                              Array<PetscInt>&      o_nnz);

private:
  MPI_Comm comm_{PETSC_COMM_SELF}; ///< Communicator used by the matrix.
  Mat      mat_{nullptr};          ///< Owned PETSc matrix handle.
  Index    rows_{0};               ///< Global row count.
  Index    cols_{0};               ///< Global column count.
};

} // namespace linalg
} // namespace femx
