#pragma once

#include <petscvec.h>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::linalg::detail
{

void check(PetscErrorCode error, const char* operation);

void checkMPI(int error, const char* operation);

void checkInit();

PetscErrorCode copyFromPETSc(Vec source, HostVector<Real>& destination);

PetscErrorCode copyToPETSc(HostVectorView<const Real> source,
                           Vec                        destination);

} // namespace femx::linalg::detail
