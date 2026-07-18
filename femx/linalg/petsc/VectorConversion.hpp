#pragma once

#include <petscvec.h>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace linalg
{
namespace detail
{

inline PetscErrorCode copyFromPETSc(Vec input, HostVector& output)
{
  PetscInt size = 0;
  PetscCall(VecGetSize(input, &size));
  output.resize(static_cast<Index>(size));

  VecScatter scatter = nullptr;
  Vec        all     = nullptr;
  PetscCall(VecScatterCreateToAll(input, &scatter, &all));
  PetscCall(VecScatterBegin(scatter,
                            input,
                            all,
                            INSERT_VALUES,
                            SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,
                          input,
                          all,
                          INSERT_VALUES,
                          SCATTER_FORWARD));

  const PetscScalar* vals = nullptr;
  PetscCall(VecGetArrayRead(all, &vals));
  for (PetscInt i = 0; i < size; ++i)
  {
    output[static_cast<Index>(i)] = PetscRealPart(vals[i]);
  }
  PetscCall(VecRestoreArrayRead(all, &vals));

  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&all));
  return PETSC_SUCCESS;
}

inline PetscErrorCode copyToPETSc(const HostVector& input, Vec output)
{
  PetscInt size = 0;
  PetscCall(VecGetSize(output, &size));
  if (input.size() != static_cast<Index>(size))
  {
    return PETSC_ERR_ARG_SIZ;
  }

  PetscInt begin = 0;
  PetscInt end   = 0;
  PetscCall(VecGetOwnershipRange(output, &begin, &end));

  PetscScalar* vals = nullptr;
  PetscCall(VecGetArray(output, &vals));
  for (PetscInt i = begin; i < end; ++i)
  {
    vals[i - begin] =
        static_cast<PetscScalar>(input[static_cast<Index>(i)]);
  }
  PetscCall(VecRestoreArray(output, &vals));
  return PETSC_SUCCESS;
}

} // namespace detail
} // namespace linalg
} // namespace femx
