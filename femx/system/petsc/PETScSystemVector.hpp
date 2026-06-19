#pragma once

#include <petscvec.h>

#include <stdexcept>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemVector.hpp>
#include <femx/system/petsc/VectorConversion.hpp>

namespace femx
{
namespace system
{

/** @brief PETSc-backed implementation of SystemVector. */
class PETScSystemVector final : public SystemVector
{
public:
  explicit PETScSystemVector(MPI_Comm comm = PETSC_COMM_SELF)
    : comm_(comm)
  {
  }

  PETScSystemVector(const PETScSystemVector&)            = delete;
  PETScSystemVector& operator=(const PETScSystemVector&) = delete;

  ~PETScSystemVector() override
  {
    if (vec_ != nullptr)
    {
      VecDestroy(&vec_);
    }
  }

  Index size() const override
  {
    return size_;
  }

  Vec vec() const
  {
    if (vec_ == nullptr)
    {
      throw std::runtime_error("PETScSystemVector is not initialized");
    }
    return vec_;
  }

  MPI_Comm comm() const
  {
    return comm_;
  }

  void resize(Index size) override
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
    const PetscInt local_size =
        comm_size == 1 ? static_cast<PetscInt>(size_) : PETSC_DECIDE;

    check(VecCreate(comm_, &vec_), "VecCreate");
    check(VecSetSizes(vec_, local_size, static_cast<PetscInt>(size_)),
          "VecSetSizes");
    check(VecSetFromOptions(vec_), "VecSetFromOptions");
    setZero();
  }

  void setZero() override
  {
    if (vec_ == nullptr)
    {
      return;
    }
    check(VecZeroEntries(vec_), "VecZeroEntries");
  }

  void set(Index row, Real value) override
  {
    setValue(row, value, INSERT_VALUES);
  }

  void add(Index row, Real value) override
  {
    setValue(row, value, ADD_VALUES);
  }

  void addAtomic(Index row, Real value) override
  {
    add(row, value);
  }

  void addValues(const PetscInt*     rows,
                 Index               count,
                 const Vector<Real>& values)
  {
    if (values.size() != count)
    {
      throw std::runtime_error(
          "PETScSystemVector local values size does not match dofs");
    }
    addValues(rows, count, values.data());
  }

  void addValues(const PetscInt* rows,
                 Index           count,
                 const Real*     values)
  {
    if (vec_ == nullptr)
    {
      throw std::runtime_error("PETScSystemVector is not initialized");
    }
    check(VecSetValues(vec_,
                       static_cast<PetscInt>(count),
                       rows,
                       values,
                       ADD_VALUES),
          "VecSetValues");
  }

  void finalize() override
  {
    if (vec_ == nullptr)
    {
      throw std::runtime_error("PETScSystemVector is not initialized");
    }
    check(VecAssemblyBegin(vec_), "VecAssemblyBegin");
    check(VecAssemblyEnd(vec_), "VecAssemblyEnd");
  }

  void copyOwnedFrom(const Vector<Real>& values)
  {
    if (values.size() != size())
    {
      throw std::runtime_error("PETScSystemVector copy size mismatch");
    }
    PetscInt begin = 0;
    PetscInt end   = 0;
    check(VecGetOwnershipRange(vec(), &begin, &end), "VecGetOwnershipRange");

    PetscScalar* data = nullptr;
    check(VecGetArray(vec(), &data), "VecGetArray");
    for (PetscInt i = begin; i < end; ++i)
    {
      data[i - begin] = static_cast<PetscScalar>(values[static_cast<Index>(i)]);
    }
    check(VecRestoreArray(vec(), &data), "VecRestoreArray");
  }

  void copyToAll(Vector<Real>& values) const
  {
    check(detail::copyFromPETSc(vec(), values), "copyFromPETSc");
  }

private:
  void setValue(Index row, Real value, InsertMode mode)
  {
    if (vec_ == nullptr)
    {
      throw std::runtime_error("PETScSystemVector is not initialized");
    }
    check(VecSetValue(vec_,
                      static_cast<PetscInt>(row),
                      static_cast<PetscScalar>(value),
                      mode),
          "VecSetValue");
  }

  static void checkInitialized()
  {
    PetscBool initialized = PETSC_FALSE;
    check(PetscInitialized(&initialized), "PetscInitialized");
    if (initialized != PETSC_TRUE)
    {
      throw std::runtime_error("PETScSystemVector requires initialized PETSc");
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

private:
  MPI_Comm comm_{PETSC_COMM_SELF};
  Vec      vec_{nullptr};
  Index    size_{0};
};

} // namespace system
} // namespace femx
