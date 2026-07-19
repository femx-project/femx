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

/**
 * @brief User-facing PETSc KSP options used by KspLinearSolver.
 */
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
 * @brief PETSc KSP adapter for linalg::LinearSolver.
 *
 * KspLinearSolver accepts PETSc-native matrices through `PetscBackend`. The
 * solver options can be set programmatically and optionally overridden by
 * PETSc's options database.
 */
class KspLinearSolver final : public LinearSolver<PetscBackend>
{
public:
  explicit KspLinearSolver(MPI_Comm comm = PETSC_COMM_SELF);

  KspLinearSolver(const KspLinearSolver&)            = delete;
  KspLinearSolver& operator=(const KspLinearSolver&) = delete;

  ~KspLinearSolver() override;

  KspOptions& opts();

  const KspOptions& opts() const;

  void solve(const PETScOperator& mat,
             const HostVector&    rhs,
             HostVector&          sol,
             PetscContext&        ctx) override;

  void solveT(const PETScOperator& mat,
              const HostVector&    rhs,
              HostVector&          sol,
              PetscContext&        ctx) override;

  KSPConvergedReason convergedReason() const;

  PetscInt its() const;

  PetscReal rnorm() const;

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace linalg
} // namespace femx
