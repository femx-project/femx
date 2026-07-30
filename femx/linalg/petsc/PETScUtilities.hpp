#pragma once

#include <petscvec.h>

#include <memory>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>

namespace femx::linalg
{

class PETScPartition;

namespace detail
{

void check(PetscErrorCode error, const char* operation);

void checkMPI(int error, const char* operation);

void checkInit();

PetscErrorCode copyFromPETSc(
    Vec                                          source,
    HostVector<Real>&                            destination,
    const std::shared_ptr<const PETScPartition>& partition = {});

PetscErrorCode copyToPETSc(
    HostVectorView<const Real>                   source,
    Vec                                          destination,
    const std::shared_ptr<const PETScPartition>& partition = {});

} // namespace detail
} // namespace femx::linalg
