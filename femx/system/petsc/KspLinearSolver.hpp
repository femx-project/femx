#pragma once

#include <petscksp.h>

#include <memory>
#include <string>

#include <femx/core/Types.hpp>
#include <femx/system/LinearSolver.hpp>

namespace femx
{
namespace system
{

class PETScSystemMatrix;
class PETScSystemVector;

struct KspOptions
{
  std::string type    = KSPGMRES;
  std::string pc_type = PCNONE;

  real_type  rtol    = 1.0e-10;
  real_type  atol    = 1.0e-50;
  real_type  dtol    = 1.0e5;
  index_type max_its = 1000;
  index_type restart = 0;

  bool nonzero_guess = false;
  bool use_opts_db   = true;
};

/** @brief PETSc KSP adapter for system::LinearSolver. */
class KspLinearSolver final : public LinearSolver
{
public:
  explicit KspLinearSolver(MPI_Comm comm = PETSC_COMM_SELF);

  KspLinearSolver(const KspLinearSolver&)            = delete;
  KspLinearSolver& operator=(const KspLinearSolver&) = delete;

  ~KspLinearSolver() override;

  KspOptions& options();

  const KspOptions& options() const;

  void solve(const LinearOperator& op,
             const Vector&         rhs,
             Vector&               out) override;

  void solveT(const LinearOperator& op,
              const Vector&         rhs,
              Vector&               out) override;

  void solve(const PETScSystemMatrix& op,
             const PETScSystemVector& rhs,
             PETScSystemVector&       out);

  void solveT(const PETScSystemMatrix& op,
              const PETScSystemVector& rhs,
              PETScSystemVector&       out);

  KSPConvergedReason convergedReason() const;

  PetscInt its() const;

  PetscReal rnorm() const;

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace system
} // namespace femx
