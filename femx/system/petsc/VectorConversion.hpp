#pragma once

#include <petscvec.h>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace system
{
namespace detail
{

inline PetscErrorCode copyFromPETSc(Vec input, Vector<Real>& output)
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

  const PetscScalar* values = nullptr;
  PetscCall(VecGetArrayRead(all, &values));
  for (PetscInt i = 0; i < size; ++i)
  {
    output[static_cast<Index>(i)] = PetscRealPart(values[i]);
  }
  PetscCall(VecRestoreArrayRead(all, &values));

  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&all));
  return PETSC_SUCCESS;
}

inline PetscErrorCode copyToPETSc(const Vector<Real>& input, Vec output)
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

  PetscScalar* values = nullptr;
  PetscCall(VecGetArray(output, &values));
  for (PetscInt i = begin; i < end; ++i)
  {
    values[i - begin] =
        static_cast<PetscScalar>(input[static_cast<Index>(i)]);
  }
  PetscCall(VecRestoreArray(output, &values));
  return PETSC_SUCCESS;
}

} // namespace detail
} // namespace system
} // namespace femx
