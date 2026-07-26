#pragma once

#include <femx/linalg/Context.hpp>
#include <femx/linalg/Jacobian.hpp>

namespace femx::linalg
{

/**
 * @brief Own a valid context, Jacobian, and solver combination.
 *
 * @tparam Space - Vector storage memory space.
 */
template <MemorySpace Space>
class LinearSystem
{
public:
  using Vector    = femx::Vector<Space, Real>;
  using ConstView = VectorView<Space, const Real>;

  virtual ~LinearSystem() = default;

  /** @brief Return the system-owned execution context. */
  virtual Context<Space>& context() noexcept = 0;

  /** @brief Return the system-owned Jacobian. */
  virtual Jacobian<Space>& jacobian() noexcept = 0;

  /**
   * @brief Solve the assembled system.
   *
   * @param[in] rhs - Right-hand side view.
   * @param[out] solution - Solution vector.
   */
  virtual void solve(ConstView rhs, Vector& solution) = 0;

  /**
   * @brief Solve the transposed assembled system.
   *
   * @param[in] rhs - Right-hand side view.
   * @param[out] solution - Solution vector.
   */
  virtual void solveT(ConstView rhs, Vector& solution) = 0;
};

} // namespace femx::linalg
