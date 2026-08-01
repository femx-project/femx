#pragma once

#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Define the CSR linear solve contract for one memory space.
 */
template <MemorySpace Space>
class LinearSolver
{
public:
  using Matrix           = CsrMatrix<Space>;
  using Vector           = femx::Vector<Space, Real>;
  using ExecutionContext = Context<Space>;

  virtual ~LinearSolver() = default;

  /**
   * @brief Solve `mat * x = rhs`.
   *
   * @param[in]     mat - System matrix.
   * @param[in]     rhs - Right-hand side vector.
   * @param[in,out] x   - Initial guess replaced by the solution.
   * @param[in]     ctx - Execution context.
   */
  virtual void solve(const Matrix&     mat,
                     const Vector&     rhs,
                     Vector&           x,
                     ExecutionContext& ctx) = 0;

  /**
   * @brief Solve `mat^T * x = rhs`.
   *
   * @param[in]     mat - System matrix.
   * @param[in]     rhs - Right-hand side vector.
   * @param[in,out] x   - Initial guess replaced by the solution.
   * @param[in]     ctx - Execution context.
   */
  virtual void solveT(const Matrix&     mat,
                      const Vector&     rhs,
                      Vector&           x,
                      ExecutionContext& ctx) = 0;
};

} // namespace femx::linalg
