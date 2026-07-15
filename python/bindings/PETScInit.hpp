#pragma once

#ifdef FEMX_HAS_PETSC
#include <petscsys.h>

#include <stdexcept>

namespace femx
{
namespace python
{

inline void initializePETSc()
{
  PetscBool finalized = PETSC_FALSE;
  if (PetscFinalized(&finalized) != PETSC_SUCCESS)
  {
    throw std::runtime_error("PetscFinalized failed");
  }
  if (finalized == PETSC_TRUE)
  {
    throw std::runtime_error("PETSc has already been finalized");
  }

  PetscBool initialized = PETSC_FALSE;
  if (PetscInitialized(&initialized) != PETSC_SUCCESS)
  {
    throw std::runtime_error("PetscInitialized failed");
  }
  if (initialized != PETSC_TRUE
      && PetscInitializeNoArguments() != PETSC_SUCCESS)
  {
    throw std::runtime_error("PetscInitializeNoArguments failed");
  }
}

} // namespace python
} // namespace femx
#endif
