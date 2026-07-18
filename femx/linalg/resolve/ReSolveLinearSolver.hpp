#pragma once

#include <memory>
#include <string>

#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>

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

  std::string gram_schmidt = "cgs2";  ///< Krylov orthogonalization method.
  std::string sketching    = "count"; ///< Sketching method for randomized Krylov variants.
  std::string pc_side      = "right"; ///< Side on which to apply preconditioning.

  Index max_its  = 1000;   ///< Maximum Krylov iterations.
  Index restart  = 200;    ///< Krylov restart length.
  Real  rtol     = 1.0e-8; ///< Relative residual tolerance.
  bool  flexible = true;   ///< Enable flexible Krylov methods.
};

/**
 * @brief ReSolve adapter for Host and Device sparse linear solves.
 *
 * Host operations retain the LinearSolver interface used by CPU state and
 * inverse workflows. Device overloads bind femx CUDA storage directly and do
 * not stage matrices or vectors through Host memory. Host and CUDA resources
 * are initialized independently on first use.
 */
class ReSolveLinearSolver final : public LinearSolver
{
public:
  /** @brief Create a lazy ReSolve linear solver with shared defaults. */
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

  /**
   * @brief Bind a Device CSR matrix without copying its graph or values.
   *
   * The matrix, graph, and their allocations must remain alive and unmoved
   * until another Device matrix is bound or the solver is destroyed.
   */
  void setOperator(const DeviceCsrMatrix& mat);

  /**
   * @brief Solve the bound Device system without Host staging.
   *
   * `rhs`, `sol`, and the bound matrix values must use distinct storage.
   */
  void solve(const DeviceVector& rhs,
             DeviceVector&       sol,
             CudaContext&        ctx);

  ReSolveLinearSolver(const ReSolveLinearSolver&)            = delete;
  ReSolveLinearSolver& operator=(const ReSolveLinearSolver&) = delete;

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace linalg
} // namespace femx
