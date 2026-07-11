#pragma once

#include <memory>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/common/Workspace.hpp>
#include <femx/linalg/LinearSolver.hpp>

namespace femx
{
class CsrMatrix;
template <typename T>
class Vector;

namespace linalg
{

/**
 * @brief ReSolve solver configuration used by ReSolveLinearSolver.
 */
struct ReSolveOptions
{
  std::string factor   = "none";   ///< Factorization method.
  std::string refactor = "none";   ///< Refactorization method.
  std::string solve    = "fgmres"; ///< Linear solve method.
  std::string precond  = "ilu0";   ///< Preconditioner method.
  std::string ir       = "none";   ///< Iterative-refinement method.

  std::string gram_schmidt = "cgs2";  ///< Krylov orthogonalization method.
  std::string sketching    = "count"; ///< Sketching method for randomized Krylov variants.

  Index max_its  = 1000;   ///< Maximum Krylov iterations.
  Index restart  = 200;    ///< Krylov restart length.
  Real  rtol     = 1.0e-8; ///< Relative residual tolerance.
  bool  flexible = true;   ///< Enable flexible Krylov methods.
};

/**
 * @brief ReSolve adapter for femx sparse linear solves.
 *
 * The adapter accepts CsrAssemblyMatrix operators and can run on the
 * configured ReSolve CPU or CUDA backend.  It implements both forward and
 * transpose solves for use in state and adjoint workflows.
 */
class ReSolveLinearSolver final : public LinearSolver
{
public:
  /** @brief Create a ReSolve linear solver for the given workspace. */
  explicit ReSolveLinearSolver(WorkspaceType workspace_type);

  /** @brief Create a ReSolve linear solver with explicit options. */
  ReSolveLinearSolver(WorkspaceType  workspace_type,
                      ReSolveOptions opts);

  /** @brief Destroy the solver and owned ReSolve resources. */
  ~ReSolveLinearSolver() override;

  /** @brief Solve op x = rhs for a CsrAssemblyMatrix operator. */
  void solve(const LinearOperator& op,
             const Vector<Real>&   rhs,
             Vector<Real>&         out) override;

  /** @brief Solve op^T x = rhs for a CsrAssemblyMatrix operator. */
  void solveT(const LinearOperator& op,
              const Vector<Real>&   rhs,
              Vector<Real>&         out) override;

  /** @brief Set the system matrix used by subsequent solves. */
  void setOperator(const CsrMatrix& A);

  /** @brief Select or update the preconditioner method. */
  void setPreconditioner(const std::string& method);

  /** @brief Solve A x = b using the current operator. */
  void solve(const Vector<Real>& b, Vector<Real>& x);

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace linalg
} // namespace femx
