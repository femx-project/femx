#pragma once

#include <petscsys.h>

#include <stdexcept>
#include <string>

namespace femx::inverse
{

inline void checkPetsc(PetscErrorCode     ierr,
                       const std::string& action)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        action + " failed with PETSc error code " + std::to_string(ierr));
  }
}

} // namespace femx::inverse
