#pragma once

#include <femx/linalg/Backend.hpp>

namespace femx::linalg
{

/**
 * @brief Temporary Host backend tag for PETSc state instantiations.
 *
 * PETSc behavior is selected by the concrete Jacobian and linear system.
 */
struct PetscBackend
{
  static constexpr MemorySpace space = MemorySpace::Host;

  using Vec       = HostVector<Real>;
  using VecView   = HostVectorView<Real>;
  using ConstView = HostVectorView<const Real>;
  using Mat       = Jacobian<MemorySpace::Host>;
  using Pattern   = HostCsrPattern;
};

static_assert(is_backend_v<PetscBackend>,
              "PetscBackend does not satisfy the backend contract");

} // namespace femx::linalg
