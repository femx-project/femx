#pragma once

#include <memory>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>

namespace femx
{
namespace linalg
{

/**
 * @brief ReSolve configuration shared by host and CUDA solver adapters.
 */
struct ReSolveOptions
{
  std::string factor   = "none";   ///< Factorization method.
  std::string refactor = "none";   ///< Refactorization method.
  std::string solve    = "fgmres"; ///< Linear solve method.
  std::string precond  = "ilu0";   ///< Preconditioner method.
  std::string ir       = "none";   ///< Iterative-refinement method.

  std::string gram_schmidt        = "cgs2";  ///< Krylov orthogonalization method.
  std::string sketching           = "count"; ///< Sketching method for randomized Krylov variants.
  std::string preconditioner_side = "right"; ///< Side on which to apply preconditioning.

  Index max_its  = 1000;   ///< Maximum Krylov iterations.
  Index restart  = 200;    ///< Krylov restart length.
  Real  rtol     = 1.0e-8; ///< Relative residual tolerance.
  bool  flexible = true;   ///< Enable flexible Krylov methods.
};

/**
 * @brief ReSolve adapter for femx sparse linear solves.
 *
 * The host adapter accepts MapCsrMatrix operators and implements both forward
 * and transpose solves for use in state and adjoint workflows.
 */
class ReSolveLinearSolver final : public LinearSolver
{
public:
  /** @brief Create a CPU ReSolve linear solver. */
  ReSolveLinearSolver();

  /** @brief Create a ReSolve linear solver with explicit options. */
  explicit ReSolveLinearSolver(ReSolveOptions opts);

  /** @brief Destroy the solver and owned ReSolve resources. */
  ~ReSolveLinearSolver() override;

  /** @brief Solve op x = rhs for a MapCsrMatrix operator. */
  void solve(const LinearOperator& op,
             const HostVector&     rhs,
             HostVector&           out) override;

  /** @brief Solve op^T x = rhs for a MapCsrMatrix operator. */
  void solveT(const LinearOperator& op,
              const HostVector&     rhs,
              HostVector&           out) override;

  /** @brief Set the system matrix used by subsequent solves. */
  void setOperator(const HostCsrMatrix& A);

  /** @brief Solve A x = b using the current operator. */
  void solve(const HostVector& b, HostVector& x);

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace linalg
} // namespace femx
