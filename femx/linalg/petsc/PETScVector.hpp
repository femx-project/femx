#pragma once

#include <petscvec.h>

#include <femx/common/Types.hpp>

namespace femx
{
namespace linalg
{

/**
 * @brief PETSc-backed mutable vector assembly target.
 *
 * PETScVector owns a PETSc Vec and provides assembly, ownership, and copy
 * helpers used by PETSc matrix assembly and solvers.
 */
class PETScVector final
{
public:
  explicit PETScVector(MPI_Comm comm = PETSC_COMM_SELF);

  PETScVector(const PETScVector&)            = delete;
  PETScVector& operator=(const PETScVector&) = delete;

  ~PETScVector();

  Index    size() const;
  Vec      vec() const;
  MPI_Comm comm() const;

  PetscInt ownershipBegin() const;
  PetscInt ownershipEnd() const;
  PetscInt localSize() const;

  void resize(Index size);
  void setZero();

  void set(Index row, Real value);
  void add(Index row, Real value);
  void addAtomic(Index row, Real value);

  void addValues(const PetscInt*   rows,
                 Index             count,
                 const HostVector& vals);

  void addValues(const PetscInt* rows,
                 Index           count,
                 const Real*     vals);

  void finalize();

  void copyOwnedFrom(const HostVector& vals);
  void copyToAll(HostVector& vals) const;

private:
  void setValue(Index row, Real value, InsertMode mode);

  static void checkInit();
  static void check(PetscErrorCode ierr, const char* operation);
  static void checkMPI(int ierr, const char* operation);

private:
  MPI_Comm comm_{PETSC_COMM_SELF};
  Vec      vec_{nullptr};
  Index    size_{0};
};

} // namespace linalg
} // namespace femx
