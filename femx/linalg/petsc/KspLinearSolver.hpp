#pragma once

#include <petscksp.h>

#include <memory>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/petsc/PETScBackend.hpp>

namespace femx
{
namespace linalg
{

class PETScOperator;

/** @brief Configure the PETSc KSP linear solver. */
struct KspOptions
{
  std::string type = KSPGMRES; ///< PETSc KSP type.
  std::string pc_type =
      PCILU; ///< PETSc PC type; block Jacobi with ILU in parallel.

  Real  rtol          = 1.0e-8;  ///< Relative residual tolerance.
  Real  atol          = 1.0e-50; ///< Absolute tolerance; disabled by default.
  Real  dtol          = 1.0e5;   ///< Divergence tolerance.
  Index max_its       = 5000;    ///< Maximum KSP iterations.
  Index restart       = 200;     ///< GMRES restart length.
  Index factor_levels = 0;       ///< ILU factor fill level.

  bool nonzero_guess = false; ///< Use the input vector as an initial guess.
  bool use_opts_db   = true;  ///< Allow PETSc options-database overrides.
};

/**
 * @brief Solve PETSc systems through the `LinearSolver` interface.
 *
 * KspLinearSolver accepts PETSc-native matrices through `PetscBackend`. The
 * solver options can be set programmatically and optionally overridden by
 * PETSc's options database.
 */
class KspLinearSolver final : public LinearSolver<PetscBackend>
{
public:
  /**
   * @brief Construct a KSP solver on a communicator.
   *
   * @param[in] comm - PETSc communicator.
   */
  explicit KspLinearSolver(MPI_Comm comm = PETSC_COMM_SELF);

  KspLinearSolver(const KspLinearSolver&) = delete;

  KspLinearSolver& operator=(const KspLinearSolver&) = delete;

  ~KspLinearSolver() override;

  /**
   * @brief Return mutable solver options.
   *
   * @return Solver configuration.
   */
  KspOptions& opts();

  /**
   * @brief Return solver options.
   *
   * @return Read-only solver configuration.
   */
  const KspOptions& opts() const;

  /**
   * @brief Solve `mat * sol = rhs`.
   *
   * @param[in] mat - Square PETSc system matrix.
   * @param[in] rhs - Replicated Host right-hand side.
   * @param[in,out] sol - Initial guess replaced by the replicated solution.
   * @param[in] ctx - PETSc execution context.
   * @throws std::runtime_error - If inputs are invalid, PETSc reports an error,
   * or the solver does not converge.
   */
  void solve(const PETScOperator& mat,
             const HostVector&    rhs,
             HostVector&          sol,
             PetscContext&        ctx) override;

  /**
   * @brief Solve `mat^T * sol = rhs`.
   *
   * @param[in] mat - Square PETSc system matrix.
   * @param[in] rhs - Replicated Host right-hand side.
   * @param[in,out] sol - Initial guess replaced by the replicated solution.
   * @param[in] ctx - PETSc execution context.
   * @throws std::runtime_error - If inputs are invalid, PETSc reports an error,
   * or the solver does not converge.
   */
  void solveT(const PETScOperator& mat,
              const HostVector&    rhs,
              HostVector&          sol,
              PetscContext&        ctx) override;

  /**
   * @brief Return the most recent KSP convergence reason.
   *
   * @return PETSc convergence reason.
   */
  KSPConvergedReason convergedReason() const;

  /**
   * @brief Return the most recent iteration count.
   *
   * @return Number of KSP iterations.
   */
  PetscInt its() const;

  /**
   * @brief Return the most recent residual norm.
   *
   * @return KSP residual norm.
   */
  PetscReal rnorm() const;

private:
  class Impl;

  std::unique_ptr<Impl> impl_; ///< Owned implementation state.
};

} // namespace linalg
} // namespace femx
