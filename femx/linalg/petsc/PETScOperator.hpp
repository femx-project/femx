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
 * @brief PETSc-backed assembly matrix.
 *
 * PETScOperator owns a PETSc Mat and provides assembly, preallocation,
 * row elimination, and matrix-vector application helpers.
 */
class PETScOperator final
{
public:
  /** @brief Create an empty PETSc matrix on `comm`. */
  explicit PETScOperator(MPI_Comm comm = PETSC_COMM_SELF);

  PETScOperator(const PETScOperator&)            = delete;
  PETScOperator& operator=(const PETScOperator&) = delete;

  /** @brief Destroy the owned PETSc matrix. */
  ~PETScOperator();

  /** @brief Return the global row count. */
  Index rows() const;
  /** @brief Return the global column count. */
  Index cols() const;

  /** @brief Return the borrowed PETSc matrix handle. */
  Mat      mat() const;
  /** @brief Return the communicator used by the matrix. */
  MPI_Comm comm() const;

  /** @brief Allocate an AIJ matrix with fallback preallocation. */
  void resize(Index rows, Index cols);
  /** @brief Allocate from an exact host CSR graph. */
  void resize(const HostCsrGraph& graph);

  /** @brief Zero all numeric entries while retaining sparsity. */
  void setZero();
  /** @brief Replace one global entry before finalize(). */
  void set(Index row, Index col, Real val);
  /** @brief Add a block using femx global-index arrays. */
  void addBlock(const Array<Index>& rows,
                const Array<Index>& cols,
                const DenseMatrix&  mat_e);

  /** @brief Complete PETSc matrix assembly. */
  void finalize();
  void replaceRows(const Array<Index>& rows, Real diag);

  /** @brief Apply the matrix to a replicated host vector. */
  void apply(HostConstVectorView dir, HostVector& out) const;
  /** @brief Apply the transpose to a replicated host vector. */
  void applyT(HostConstVectorView dir, HostVector& out) const;

private:
  class ScopedVec
  {
  public:
    ~ScopedVec();

    Vec  get() const;
    Vec* put();

  private:
    Vec vec_{nullptr};
  };

  void createVec(Index size, ScopedVec& out) const;
  void addBlock(const PetscInt* rows,
                Index           num_rows,
                const PetscInt* cols,
                Index           num_cols,
                const Real*     vals);
  void zeroRows(const Array<Index>& rows, Real diag);

  static void computePrealloc(const HostCsrGraph& graph,
                              PetscInt            begin,
                              PetscInt            end,
                              Array<PetscInt>&    d_nnz,
                              Array<PetscInt>&    o_nnz);

private:
  MPI_Comm comm_{PETSC_COMM_SELF};
  Mat      mat_{nullptr};
  Index    rows_{0};
  Index    cols_{0};
};

} // namespace linalg
} // namespace femx
