#pragma once

#include <memory>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/LinearSolver.hpp>

namespace femx
{
namespace linalg
{

/**
 * @brief Configure Host and CUDA ReSolve adapters.
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
 * @brief Solve Host and Device sparse systems with ReSolve.
 *
 * Host operations retain the LinearSolver interface used by CPU state and
 * inverse workflows. Device overloads bind femx CUDA storage directly and do
 * not stage matrices or vectors through Host memory. Host and CUDA resources
 * are initialized independently on first use.
 */
class ReSolveLinearSolver final
  : public LinearSolver<MemorySpace::Host>,
    public LinearSolver<MemorySpace::Device>
{
public:
  /**
   * @brief Construct a solver with default ReSolve options.
   */
  ReSolveLinearSolver();

  /**
   * @brief Construct a solver with ReSolve configuration options.
   *
   * @param[in] opts - Solver configuration.
   */
  explicit ReSolveLinearSolver(ReSolveOptions opts);

  ~ReSolveLinearSolver() override;

  /**
   * @brief Solve `mat * x = rhs` on Host.
   *
   * @param[in]  mat - Square Host system matrix.
   * @param[in]  rhs - Host right-hand side vector.
   * @param[out] x - Host solution vector.
   * @param[in]  ctx - CPU execution context.
   * @throws - If the inputs or solver configuration are
   * invalid, initialization fails, or ReSolve reports an error.
   */
  void solve(const HostCsrMatrix&        mat,
             const HostVector<Real>&     rhs,
             HostVector<Real>&           x,
             Context<MemorySpace::Host>& ctx) override;

  /**
   * @brief Solve `mat^T * x = rhs` on Host.
   *
   * @param[in]  mat - Square Host system matrix.
   * @param[in]  rhs - Host right-hand side vector.
   * @param[out] x - Host solution vector.
   * @param[in]  ctx - CPU execution context.
   * @throws - If the inputs or solver configuration are
   * invalid, initialization fails, or ReSolve reports an error.
   */
  void solveT(const HostCsrMatrix&        mat,
              const HostVector<Real>&     rhs,
              HostVector<Real>&           x,
              Context<MemorySpace::Host>& ctx) override;

  /**
   * @brief Solve `mat * x = rhs` on Device without Host staging.
   *
   * @param[in]  mat - Square Device system matrix.
   * @param[in]  rhs - Device right-hand side vector.
   * @param[out] x - Device solution vector.
   * @param[in]  ctx - CUDA execution context.
   * @throws - If the inputs, CUDA support, or solver
   * configuration are invalid, or ReSolve reports an error.
   */
  void solve(const DeviceCsrMatrix&        mat,
             const DeviceVector<Real>&     rhs,
             DeviceVector<Real>&           x,
             Context<MemorySpace::Device>& ctx) override;

  /**
   * @brief Solve `mat^T * x = rhs` on Device.
   *
   * @param[in]  mat - Square Device system matrix.
   * @param[in]  rhs - Device right-hand side vector.
   * @param[out] x - Device solution vector.
   * @param[in]  ctx - CUDA execution context.
   * @throws - If the inputs, CUDA support, or solver
   * configuration are invalid, or ReSolve reports an error.
   */
  void solveT(const DeviceCsrMatrix&        mat,
              const DeviceVector<Real>&     rhs,
              DeviceVector<Real>&           x,
              Context<MemorySpace::Device>& ctx) override;

  ReSolveLinearSolver(const ReSolveLinearSolver&) = delete;

  ReSolveLinearSolver& operator=(const ReSolveLinearSolver&) = delete;

private:
  class Impl;

  std::unique_ptr<Impl> impl_; ///< Owned implementation state.
};

} // namespace linalg
} // namespace femx
