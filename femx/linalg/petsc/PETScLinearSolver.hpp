#pragma once

#include <petscksp.h>

#include <memory>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>

namespace femx
{
namespace linalg
{

class PETScMatrix;

/**
 * @brief Configure the PETSc KSP linear solver.
 */
struct KspOptions
{
  std::string type    = KSPGMRES; ///< PETSc KSP type.
  std::string pc_type = PCILU;    ///< PETSc PC type; block Jacobi with ILU in parallel.

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
 * @brief Own a PETSc KSP solver for PETSc-native matrices.
 *
 * The solver options can be set programmatically and optionally overridden by
 * PETSc's options database.
 */
class PETScLinearSolver final
{
public:
  /**
   * @brief Construct a KSP solver on a communicator.
   *
   * @param[in] comm - PETSc communicator.
   */
  explicit PETScLinearSolver(MPI_Comm comm = PETSC_COMM_SELF);

  PETScLinearSolver(const PETScLinearSolver&) = delete;

  PETScLinearSolver& operator=(const PETScLinearSolver&) = delete;

  ~PETScLinearSolver();

  /**
   * @brief Return mutable solver options.
   */
  KspOptions& opts();

  /**
   * @brief Return read-only solver options.
   */
  const KspOptions& opts() const;

  /**
   * @brief Solve `mat * x = rhs`.
   *
   * @param[in]     mat - Square PETSc system matrix.
   * @param[in]     rhs - Replicated Host right-hand side.
   * @param[in,out] x - Initial guess replaced by the replicated solution.
   * @throws - If inputs are invalid, PETSc reports an error,
   * or the solver does not converge.
   */
  void solve(const PETScMatrix&      mat,
             const HostVector<Real>& rhs,
             HostVector<Real>&       x);

  /**
   * @brief Solve `mat^T * x = rhs`.
   *
   * @param[in]     mat - Square PETSc system matrix.
   * @param[in]     rhs - Replicated Host right-hand side.
   * @param[in,out] x - Initial guess replaced by the replicated solution.
   * @throws - If inputs are invalid, PETSc reports an error,
   * or the solver does not converge.
   */
  void solveT(const PETScMatrix&      mat,
              const HostVector<Real>& rhs,
              HostVector<Real>&       x);

  /**
   * @brief Return the most recent KSP convergence reason.
   */
  KSPConvergedReason convergedReason() const;

  /**
   * @brief Return the most recent KSP iteration count.
   */
  PetscInt its() const;

  /**
   * @brief Return the most recent KSP residual norm.
   */
  PetscReal rnorm() const;

private:
  class Impl;

  std::unique_ptr<Impl> impl_; ///< Owned implementation state.
};

} // namespace linalg
} // namespace femx
