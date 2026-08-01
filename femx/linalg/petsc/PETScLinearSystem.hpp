#pragma once

#include <femx/linalg/LinearSystem.hpp>
#include <femx/linalg/petsc/MpiContext.hpp>
#include <femx/linalg/petsc/PETScLinearSolver.hpp>
#include <femx/linalg/petsc/PETScSystemMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Own an MPI context, PETSc system matrix, and PETSc solver.
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

  /**
   * @copydoc LinearSystem::context()
   */
  Context<MemorySpace::Host>& context() noexcept override;

  /**
   * @copydoc LinearSystem::matrix()
   */
  SystemMatrix<MemorySpace::Host>& matrix() noexcept override;

  /**
   * @copydoc LinearSystem::solve()
   */
  void solve(VectorView rhs, Vector& x) override;

  /**
   * @copydoc LinearSystem::solveT()
   */
  void solveT(VectorView rhs, Vector& x) override;

  /**
   * @brief Return the owned PETSc solver for option configuration.
   */
  PETScLinearSolver& solver() noexcept;

  /**
   * @brief Return the owned PETSc solver for option inspection.
   */
  const PETScLinearSolver& solver() const noexcept;

private:
  MpiContext        ctx_;    ///< Owned MPI execution context.
  PETScSystemMatrix mat_;    ///< Owned PETSc system matrix.
  PETScLinearSolver solver_; ///< Owned PETSc linear solver.
  HostVector<Real>  rhs_;    ///< Host copy of the right-hand side.
};

} // namespace femx::linalg
