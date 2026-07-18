#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <femx/common/Types.hpp>
#include <femx/linalg/MatrixOperator.hpp>

namespace femx
{
class DenseMatrix;

namespace linalg
{

class PETScVector;

/**
 * @brief PETSc-backed assembly matrix.
 *
 * PETScOperator owns a PETSc Mat and provides assembly, preallocation,
 * row elimination, and matrix-vector application helpers.
 */
class PETScOperator final : public MatrixOperator
{
public:
  /** @brief Create an empty PETSc matrix on `comm`. */
  explicit PETScOperator(MPI_Comm comm = PETSC_COMM_SELF);

  PETScOperator(const PETScOperator&)            = delete;
  PETScOperator& operator=(const PETScOperator&) = delete;

  /** @brief Destroy the owned PETSc matrix. */
  ~PETScOperator() override;

  /** @brief Return the global row count. */
  Index numRows() const override;
  /** @brief Return the global column count. */
  Index numCols() const override;

  /** @brief Return the borrowed PETSc matrix handle. */
  Mat      mat() const;
  /** @brief Return the communicator used by the matrix. */
  MPI_Comm comm() const;

  /** @brief Set the fallback diagonal preallocation per local row. */
  void setDefaultNnzPerRow(Index nnz);

  /** @brief Allocate an AIJ matrix with fallback preallocation. */
  void resize(Index rows, Index cols) override;
  /** @brief Allocate from an exact host CSR graph and distributed layout. */
  void resize(const HostCsrGraph& graph, const PETScVector& lyt);

  /** @brief Zero all numeric entries while retaining sparsity. */
  void setZero() override;
  /** @brief Replace one global entry before finalize(). */
  void set(Index row, Index col, Real val) override;
  /** @brief Accumulate one global entry before finalize(). */
  void add(Index row, Index col, Real val) override;
  /** @brief Accumulate one entry from a threaded assembly region. */
  void addAtomic(Index row, Index col, Real val) override;

  /** @brief Add a square dense block on one DOF list. */
  void addBlock(const PetscInt*    dofs,
                Index              num_dofs,
                const DenseMatrix& mat_e);
  /** @brief Add a square row-major block on one DOF list. */
  void addBlock(const PetscInt* dofs,
                Index           num_dofs,
                const Real*     vals);
  /** @brief Add a rectangular dense block on row and column DOF lists. */
  void addBlock(const PetscInt*    rows,
                Index              num_rows,
                const PetscInt*    cols,
                Index              num_cols,
                const DenseMatrix& mat_e);
  /** @brief Add a rectangular row-major block on row and column DOF lists. */
  void addBlock(const PetscInt* rows,
                Index           num_rows,
                const PetscInt* cols,
                Index           num_cols,
                const Real*     vals);

  /** @brief Complete PETSc matrix assembly. */
  void finalize() override;

  /** @brief Eliminate rows and columns while correcting a PETSc RHS. */
  void zeroRowsCols(const Array<Index>& rows,
                    Real                diag,
                    const PETScVector&  vals,
                    PETScVector&        rhs);
  /** @brief Replace selected rows by diagonal rows. */
  void zeroRows(const Array<Index>& rows, Real diag);

  /** @brief Apply the matrix to a replicated host vector. */
  void apply(const HostVector& dir, HostVector& out) const override;
  /** @brief Apply the transpose to a replicated host vector. */
  void applyT(const HostVector& dir, HostVector& out) const override;

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
  void setValue(Index row, Index col, Real val, InsertMode mode);

  static void checkInit();
  static void check(PetscErrorCode ierr, const char* op);
  static void checkMPI(int ierr, const char* op);

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
  Index    default_nnz_per_row_{32};
};

} // namespace linalg
} // namespace femx
