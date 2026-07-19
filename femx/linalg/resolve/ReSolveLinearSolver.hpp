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
class ReSolveLinearSolver final : public HostCsrLinearSolver,
                                  public DeviceLinearSolver
{
public:
  /** @brief Create a lazy ReSolve linear solver with shared defaults. */
  ReSolveLinearSolver();

  /** @brief Create a ReSolve linear solver with explicit options. */
  explicit ReSolveLinearSolver(ReSolveOptions opts);

  /** @brief Destroy the solver and owned ReSolve resources. */
  ~ReSolveLinearSolver() override;

  /** @brief Solve a concrete Host CSR system. */
  void solve(const HostCsrMatrix& mat,
             const HostVector&    rhs,
             HostVector&          sol,
             CpuContext&          ctx) override;

  /** @brief Solve a concrete transposed Host CSR system. */
  void solveT(const HostCsrMatrix& mat,
              const HostVector&    rhs,
              HostVector&          sol,
              CpuContext&          ctx) override;

  /** @brief Solve a concrete Device CSR system without Host staging. */
  void solve(const DeviceCsrMatrix& mat,
             const DeviceVector&    rhs,
             DeviceVector&          sol,
             CudaContext&           ctx) override;

  /** @brief Solve a concrete transposed Device CSR system on Device. */
  void solveT(const DeviceCsrMatrix& mat,
              const DeviceVector&    rhs,
              DeviceVector&          sol,
              CudaContext&           ctx) override;

  ReSolveLinearSolver(const ReSolveLinearSolver&)            = delete;
  ReSolveLinearSolver& operator=(const ReSolveLinearSolver&) = delete;

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace linalg
} // namespace femx
