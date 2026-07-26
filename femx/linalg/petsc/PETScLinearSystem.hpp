#pragma once

#include <femx/linalg/LinearSystem.hpp>
#include <femx/linalg/petsc/MpiContext.hpp>
#include <femx/linalg/petsc/PETScJacobian.hpp>
#include <femx/linalg/petsc/PETScLinearSolver.hpp>

namespace femx::linalg
{

/**
 * @brief Own an MPI context, PETSc Jacobian, and PETSc solver.
 */
class PETScLinearSystem final : public LinearSystem<MemorySpace::Host>
{
public:
  /**
   * @brief Construct a complete PETSc linear system.
   *
   * @param[in] comm - MPI communicator duplicated by the system context.
   */
  explicit PETScLinearSystem(MPI_Comm comm = PETSC_COMM_SELF);

  PETScLinearSystem(const PETScLinearSystem&)            = delete;
  PETScLinearSystem& operator=(const PETScLinearSystem&) = delete;
  PETScLinearSystem(PETScLinearSystem&&)                 = delete;
  PETScLinearSystem& operator=(PETScLinearSystem&&)      = delete;

  Context<MemorySpace::Host>&  context() noexcept override;
  Jacobian<MemorySpace::Host>& jacobian() noexcept override;
  void                         solve(ConstView rhs, Vector& solution) override;
  void                         solveT(ConstView rhs, Vector& solution) override;

  /** @brief Return the owned PETSc solver for option configuration. */
  PETScLinearSolver& solver() noexcept;

  /** @brief Return the owned PETSc solver for option inspection. */
  const PETScLinearSolver& solver() const noexcept;

private:
  MpiContext        ctx_;
  PETScJacobian     jac_;
  PETScLinearSolver solver_;
  HostVector<Real>  rhs_;
};

} // namespace femx::linalg
