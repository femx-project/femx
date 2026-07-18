#pragma once

#include <memory>

#include <femx/common/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>

namespace femx
{
namespace linalg
{

/**
 * @brief Persistent zero-copy adapter for ReSolve's CUDA workspace.
 *
 * Matrices and vectors remain owned by femx. ReSolve wrappers borrow their
 * device pointers and are reused across solves. A bound matrix, its graph, and
 * their device allocations must remain alive and unmoved until that binding is
 * replaced or the adapter is destroyed. Solve vectors must remain alive for
 * the duration of the solve and must not alias each other or matrix storage.
 * The adapter itself does not stage system data per solve; the current ReSolve
 * FGMRES implementation may still allocate internal Krylov/SpMV workspace.
 */
class ReSolveDeviceSolver
{
public:
  ReSolveDeviceSolver();
  explicit ReSolveDeviceSolver(ReSolveOptions opts);
  ~ReSolveDeviceSolver();

  ReSolveDeviceSolver(const ReSolveDeviceSolver&)            = delete;
  ReSolveDeviceSolver& operator=(const ReSolveDeviceSolver&) = delete;

  void setOperator(const DeviceCsrMatrix& mat);

  void solve(const DeviceVector& rhs,
             DeviceVector&       sol,
             CudaContext&        ctx);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace linalg
} // namespace femx
