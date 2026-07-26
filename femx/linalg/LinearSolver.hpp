#pragma once

#include <femx/linalg/Backend.hpp>
#include <femx/linalg/Context.hpp>

namespace femx::linalg
{

/** @brief Define the linear solve contract for an execution backend. */
template <class Backend>
class LinearSolver
{
  static_assert(is_backend_v<Backend>,
                "LinearSolver requires a valid backend type");

public:
  using Matrix           = typename Backend::Mat;
  using Vector           = typename Backend::Vec;
  using ExecutionContext = Context<Backend::space>;

  virtual ~LinearSolver() = default;

  /**
   * @brief Solve `mat * sol = rhs`.
   *
   * @param[in] mat - System matrix.
   * @param[in] rhs - Right-hand side vector.
   * @param[in,out] sol - Initial guess replaced by the solution.
   * @param[in] ctx - Execution context.
   */
  virtual void solve(const Matrix&     mat,
                     const Vector&     rhs,
                     Vector&           sol,
                     ExecutionContext& ctx) = 0;

  /**
   * @brief Solve `mat^T * sol = rhs`.
   *
   * @param[in] mat - System matrix.
   * @param[in] rhs - Right-hand side vector.
   * @param[in,out] sol - Initial guess replaced by the solution.
   * @param[in] ctx - Execution context.
   */
  virtual void solveT(const Matrix&     mat,
                      const Vector&     rhs,
                      Vector&           sol,
                      ExecutionContext& ctx) = 0;
};

using HostCsrLinearSolver = LinearSolver<HostCsrBackend>;
using DeviceLinearSolver  = LinearSolver<CudaCsrBackend>;

} // namespace femx::linalg
