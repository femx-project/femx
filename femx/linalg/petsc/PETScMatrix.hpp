#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/operator/MatrixOperator.hpp>
#include <femx/linalg/petsc/PETScVector.hpp>

namespace femx
{
namespace linalg
{

/** @brief PETSc-backed matrix operator and assembly target. */
class PETScMatrix final : public MatrixOperator
{
public:
  explicit PETScMatrix(MPI_Comm comm = PETSC_COMM_SELF);

  PETScMatrix(const PETScMatrix&)            = delete;
  PETScMatrix& operator=(const PETScMatrix&) = delete;

  ~PETScMatrix() override;

  Index numRows() const override;
  Index numCols() const override;

  Mat mat() const;
  MPI_Comm comm() const;

  void setDefaultNonzerosPerRow(Index count);

  void resize(Index rows, Index cols) override;
  void resize(const CsrPattern& pettern, const PETScVector& lyt);

  void setZero() override;
  void set(Index row, Index col, Real value) override;
  void add(Index row, Index col, Real value) override;
  void addAtomic(Index row, Index col, Real value) override;

  void addBlock(const PetscInt* dofs,
                Index           nd,
                const DenseMatrix& local);
  void addBlock(const PetscInt* dofs, Index nd, const Real* vals);
  void addBlock(const PetscInt* rows,
                Index           num_rows,
                const PetscInt* cols,
                Index           num_cols,
                const DenseMatrix& local);
  void addBlock(const PetscInt* rows,
                Index           num_rows,
                const PetscInt* cols,
                Index           num_cols,
                const Real*     vals);

  void finalize() override;

  void zeroRowsColumns(const Vector<Index>& rows,
                       Real                 diagonal,
                       const PETScVector&   vals,
                       PETScVector&         rhs);
  void zeroRows(const Vector<Index>& rows, Real diagonal);

  void apply(const Vector<Real>& dir, Vector<Real>& out) const override;
  void applyT(const Vector<Real>& dir, Vector<Real>& out) const override;

private:
  class ScopedVec
  {
  public:
    ~ScopedVec();

    Vec get() const;
    Vec* put();

  private:
    Vec vec_{nullptr};
  };

  void createVec(Index size, ScopedVec& out) const;
  void setValue(Index row, Index col, Real value, InsertMode mode);

  static void checkInitialized();
  static void check(PetscErrorCode ierr, const char* operation);
  static void checkMPI(int ierr, const char* operation);

  static void computePreallocation(const CsrPattern& pettern,
                                   PetscInt          begin,
                                   PetscInt          end,
                                   Vector<PetscInt>& d_nnz,
                                   Vector<PetscInt>& o_nnz);

private:
  MPI_Comm comm_{PETSC_COMM_SELF};
  Mat      mat_{nullptr};
  Index    rows_{0};
  Index    cols_{0};
  Index    default_nonzeros_per_row_{32};
};

} // namespace linalg
} // namespace femx
