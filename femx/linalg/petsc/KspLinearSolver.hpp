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

class PETScMatrixOperator;
class PETScVectorBuilder;

struct KspOptions
{
  std::string type    = KSPGMRES;
  std::string pc_type = PCNONE;

  Real  rtol    = 1.0e-10;
  Real  atol    = 1.0e-50;
  Real  dtol    = 1.0e5;
  Index max_its = 1000;
  Index restart = 0;

  bool nonzero_guess = false;
  bool use_opts_db   = true;
  bool check_finite  = false;
};

/** @brief PETSc KSP adapter for linalg::LinearSolver. */
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

  void solve(const PETScMatrixOperator& op,
             const PETScVectorBuilder&  rhs,
             PETScVectorBuilder&        out);

  void solveT(const PETScMatrixOperator& op,
              const PETScVectorBuilder&  rhs,
              PETScVectorBuilder&        out);

  KSPConvergedReason convergedReason() const;

  PetscInt its() const;

  PetscReal rnorm() const;

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace linalg
} // namespace femx
