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
   * @brief Return the system-owned MPI execution context.
   */
  Context<MemorySpace::Host>& context() noexcept override;

  /**
   * @brief Return the system-owned PETSc matrix.
   */
  SystemMatrix<MemorySpace::Host>& matrix() noexcept override;

  /**
   * @brief Solve the assembled PETSc system.
   *
   * @param[in]  rhs - Right-hand side view.
   * @param[out] x - Solution vector.
   */
  void solve(ConstView rhs, Vector& x) override;

  /**
   * @brief Solve the transposed assembled PETSc system.
   *
   * @param[in]  rhs - Right-hand side view.
   * @param[out] x - Solution vector.
   */
  void solveT(ConstView rhs, Vector& x) override;

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
