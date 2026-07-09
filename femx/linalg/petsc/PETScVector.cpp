#include <stdexcept>
#include <string>

#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/PETScVector.hpp>
#include <femx/linalg/petsc/VectorConversion.hpp>

namespace femx
{
namespace linalg
{

PETScVector::PETScVector(MPI_Comm comm)
  : comm_(comm)
{
}

PETScVector::~PETScVector()
{
  if (vec_ != nullptr)
  {
    VecDestroy(&vec_);
  }
}

Index PETScVector::size() const
{
  return size_;
}

Vec PETScVector::vec() const
{
  if (vec_ == nullptr)
  {
    throw std::runtime_error("PETScVector is not initialized");
  }
  return vec_;
}

MPI_Comm PETScVector::comm() const
{
  return comm_;
}

PetscInt PETScVector::ownershipBegin() const
{
  PetscInt begin = 0;
  PetscInt end   = 0;
  check(VecGetOwnershipRange(vec(), &begin, &end), "VecGetOwnershipRange");
  return begin;
}

PetscInt PETScVector::ownershipEnd() const
{
  PetscInt begin = 0;
  PetscInt end   = 0;
  check(VecGetOwnershipRange(vec(), &begin, &end), "VecGetOwnershipRange");
  return end;
}

PetscInt PETScVector::localSize() const
{
  PetscInt local_size = 0;
  check(VecGetLocalSize(vec(), &local_size), "VecGetLocalSize");
  return local_size;
}

void PETScVector::resize(Index size)
{
  checkInitialized();

  if (vec_ != nullptr && size_ == size)
  {
    setZero();
    return;
  }

  if (vec_ != nullptr)
  {
    check(VecDestroy(&vec_), "VecDestroy");
  }

  size_                 = size;
  PetscMPIInt comm_size = 1;
  checkMPI(MPI_Comm_size(comm_, &comm_size), "MPI_Comm_size");
  const PetscInt num_local_dofs =
      comm_size == 1 ? static_cast<PetscInt>(size_) : PETSC_DECIDE;

  check(VecCreate(comm_, &vec_), "VecCreate");
  check(VecSetSizes(vec_, num_local_dofs, static_cast<PetscInt>(size_)),
        "VecSetSizes");
  check(VecSetFromOptions(vec_), "VecSetFromOptions");
  setZero();
}

void PETScVector::setZero()
{
  if (vec_ == nullptr)
  {
    return;
  }
  check(VecZeroEntries(vec_), "VecZeroEntries");
}

void PETScVector::set(Index row, Real value)
{
  setValue(row, value, INSERT_VALUES);
}

void PETScVector::add(Index row, Real value)
{
  setValue(row, value, ADD_VALUES);
}

void PETScVector::addAtomic(Index row, Real value)
{
  add(row, value);
}

void PETScVector::addValues(const PetscInt*     rows,
                            Index               count,
                            const Vector<Real>& vals)
{
  if (vals.size() != count)
  {
    throw std::runtime_error(
        "PETScVector local values size does not match dofs");
  }
  addValues(rows, count, vals.data());
}

void PETScVector::addValues(const PetscInt* rows,
                            Index           count,
                            const Real*     vals)
{
  if (vec_ == nullptr)
  {
    throw std::runtime_error("PETScVector is not initialized");
  }
  check(VecSetValues(vec_,
                     static_cast<PetscInt>(count),
                     rows,
                     vals,
                     ADD_VALUES),
        "VecSetValues");
}

void PETScVector::finalize()
{
  if (vec_ == nullptr)
  {
    throw std::runtime_error("PETScVector is not initialized");
  }
  check(VecAssemblyBegin(vec_), "VecAssemblyBegin");
  check(VecAssemblyEnd(vec_), "VecAssemblyEnd");
}

void PETScVector::copyOwnedFrom(const Vector<Real>& vals)
{
  if (vals.size() != size())
  {
    throw std::runtime_error("PETScVector copy size mismatch");
  }

  const PetscInt begin = ownershipBegin();
  const PetscInt end   = ownershipEnd();

  PetscScalar* data = nullptr;
  check(VecGetArray(vec(), &data), "VecGetArray");
  for (PetscInt i = begin; i < end; ++i)
  {
    data[i - begin] = static_cast<PetscScalar>(vals[static_cast<Index>(i)]);
  }
  check(VecRestoreArray(vec(), &data), "VecRestoreArray");
}

void PETScVector::copyToAll(Vector<Real>& vals) const
{
  check(detail::copyFromPETSc(vec(), vals), "copyFromPETSc");
}

void PETScVector::setValue(Index row, Real value, InsertMode mode)
{
  if (vec_ == nullptr)
  {
    throw std::runtime_error("PETScVector is not initialized");
  }
  check(VecSetValue(vec_,
                    static_cast<PetscInt>(row),
                    static_cast<PetscScalar>(value),
                    mode),
        "VecSetValue");
}

void PETScVector::checkInitialized()
{
  PetscBool initialized = PETSC_FALSE;
  check(PetscInitialized(&initialized), "PetscInitialized");
  if (initialized != PETSC_TRUE)
  {
    throw std::runtime_error("PETScVector requires initialized PETSc");
  }
}

void PETScVector::check(PetscErrorCode ierr, const char* operation)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

void PETScVector::checkMPI(int ierr, const char* operation)
{
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

} // namespace linalg
} // namespace femx
