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
  explicit PETScOperator(MPI_Comm comm = PETSC_COMM_SELF);

  PETScOperator(const PETScOperator&)            = delete;
  PETScOperator& operator=(const PETScOperator&) = delete;

  ~PETScOperator() override;

  Index numRows() const override;
  Index numCols() const override;

  Mat      mat() const;
  MPI_Comm comm() const;

  void setDefaultNnzPerRow(Index nnz);

  void resize(Index rows, Index cols) override;
  void resize(const HostCsrGraph& graph, const PETScVector& lyt);

  void setZero() override;
  void set(Index row, Index col, Real val) override;
  void add(Index row, Index col, Real val) override;
  void addAtomic(Index row, Index col, Real val) override;

  void addBlock(const PetscInt*    dofs,
                Index              num_dofs,
                const DenseMatrix& mat_e);
  void addBlock(const PetscInt* dofs,
                Index           num_dofs,
                const Real*     vals);
  void addBlock(const PetscInt*    rows,
                Index              num_rows,
                const PetscInt*    cols,
                Index              num_cols,
                const DenseMatrix& mat_e);
  void addBlock(const PetscInt* rows,
                Index           num_rows,
                const PetscInt* cols,
                Index           num_cols,
                const Real*     vals);

  void finalize() override;

  void zeroRowsCols(const Array<Index>& rows,
                    Real                diag,
                    const PETScVector&  vals,
                    PETScVector&        rhs);
  void zeroRows(const Array<Index>& rows, Real diag);

  void apply(const HostVector& dir, HostVector& out) const override;
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
