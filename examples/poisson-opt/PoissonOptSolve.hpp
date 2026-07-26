#pragma once

#include <mpi.h>

#include "PoissonOptProblem.hpp"
#include <femx/linalg/LinearSystem.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx::examples::poisson_opt
{

/** @brief Return the TAO result and final controlled state. */
struct Result
{
  Report           report;             ///< Final diagnostic metrics.
  HostVector<Real> control;            ///< Optimized boundary control.
  HostVector<Real> state;              ///< State at the optimized control.
  HostVector<Real> gradient;           ///< Final reduced gradient.
  Index            iterations = 0;     ///< Number of TAO iterations.
  int              reason     = 0;     ///< PETSc/TAO convergence reason.
  bool             converged  = false; ///< Whether TAO converged.
};

/**
 * @brief Optimize with Host state and adjoint systems.
 *
 * @param[in,out] problem - Problem whose objective is prepared.
 * @param[in,out] state_solver - Host forward state solver.
 * @param[in,out] adj_system - Host adjoint linear system.
 * @param[in] comm - Communicator used by TAO.
 * @return Optimization result and final state.
 */
Result optimize(
    PoissonOptProblem&                       problem,
    state::StateSolver<MemorySpace::Host>&   state_solver,
    linalg::LinearSystem<MemorySpace::Host>& adj_system,
    MPI_Comm                                 comm);

#if defined(FEMX_HAS_CUDA)
/**
 * @brief Optimize with Device state and adjoint systems.
 *
 * @param[in,out] problem - Problem whose objective is prepared.
 * @param[in,out] state_solver - Device forward state solver.
 * @param[in,out] adj_system - Device adjoint linear system.
 * @param[in] comm - Communicator used by TAO.
 * @return Optimization result and final state.
 */
Result optimize(
    PoissonOptProblem&                         problem,
    state::StateSolver<MemorySpace::Device>&   state_solver,
    linalg::LinearSystem<MemorySpace::Device>& adj_system,
    MPI_Comm                                   comm);
#endif

} // namespace femx::examples::poisson_opt
