#pragma once

#include <petscksp.h>

#include <memory>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>

namespace femx
{
namespace linalg
{

class PETScAssemblyMatrix;
class PETScVector;

/**
 * @brief User-facing PETSc KSP options used by KspLinearSolver.
 */
struct KspOptions
{
  std::string type    = KSPGMRES; ///< PETSc KSP type.
  std::string pc_type = PCNONE;   ///< PETSc PC type.

  Real  rtol    = 1.0e-10; ///< Relative residual tolerance.
  Real  atol    = 1.0e-50; ///< Absolute residual tolerance.
  Real  dtol    = 1.0e5;   ///< Divergence tolerance.
  Index max_its = 1000;    ///< Maximum KSP iterations.
  Index restart = 0;       ///< GMRES restart length.

  bool nonzero_guess = false; ///< Use the input vector as an initial guess.
  bool use_opts_db   = true;  ///< Allow PETSc options-database overrides.
  bool check_finite  = false; ///< Check matrix/vector values before solving.
};

/**
 * @brief PETSc KSP adapter for linalg::LinearSolver.
 *
 * KspLinearSolver accepts femx linear operators and PETSc-native assembly
 * matrices. The solver options can be set programmatically and optionally
 * overridden by PETSc's options database.
 */
class KspLinearSolver final : public LinearSolver
{
public:
  explicit KspLinearSolver(MPI_Comm comm = PETSC_COMM_SELF);

  KspLinearSolver(const KspLinearSolver&)            = delete;
  KspLinearSolver& operator=(const KspLinearSolver&) = delete;

  ~KspLinearSolver() override;

  KspOptions& opts();

  const KspOptions& opts() const;

  void solve(const LinearOperator& op,
             const Vector<Real>&   rhs,
             Vector<Real>&         out) override;

  void solveT(const LinearOperator& op,
              const Vector<Real>&   rhs,
              Vector<Real>&         out) override;

  void solve(const PETScAssemblyMatrix& op,
             const PETScVector&         rhs,
             PETScVector&               out);

  void solveT(const PETScAssemblyMatrix& op,
              const PETScVector&         rhs,
              PETScVector&               out);

  KSPConvergedReason convergedReason() const;

  PetscInt its() const;

  PetscReal rnorm() const;

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace linalg
} // namespace femx
