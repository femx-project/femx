#pragma once

#include <femx/linalg/Context.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/SystemMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Own a valid context, system matrix, and solver combination.
 *
 * @tparam Space - Vector storage memory space.
 */
template <MemorySpace Space>
class LinearSystem
{
public:
  using Vector     = femx::Vector<Space, Real>;
  using VectorView = femx::VectorView<Space, const Real>;
  using Solver     = LinearSolver<Space>;

  virtual ~LinearSystem() = default;

  /**
   * @brief Return the system-owned execution context.
   */
  virtual Context<Space>& context() noexcept = 0;

  /**
   * @brief Return the system-owned matrix.
   */
  virtual SystemMatrix<Space>& matrix() noexcept = 0;

  /**
   * @brief Solve the assembled system.
   *
   * @param[in]  rhs - Right-hand side view.
   * @param[out] x   - Solution vector.
   */
  virtual void solve(VectorView rhs, Vector& x) = 0;

  /**
   * @brief Solve the transposed assembled system.
   *
   * @param[in]  rhs - Right-hand side view.
   * @param[out] x   - Solution vector.
   */
  virtual void solveT(VectorView rhs, Vector& x) = 0;
};

} // namespace femx::linalg
