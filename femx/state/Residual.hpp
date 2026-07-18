#pragma once

#include <femx/common/Types.hpp>

namespace femx
{
namespace state
{

class Linearization;

/**
 * @brief Sizes for a parameter-dependent residual problem.
 */
struct Dimensions
{
  Index num_states = 0; ///< Size of the state vector.
  Index num_param  = 0; ///< Size of the parameter/control vector.
  Index num_res    = 0; ///< Size of the residual vector.
};

/**
 * @brief Nonlinear residual R(u, m) with linearization support.
 *
 * Residual implementations define the state equation used by state solvers and
 * reduced functionals.  The linearize() method fills a Linearization object
 * with Jacobians with respect to the state and parameter vectors.
 */
class Residual
{
public:
  virtual ~Residual() = default;

  /** @brief Return state, parameter, and residual dimensions. */
  virtual Dimensions dims() const = 0;

  /** @brief Evaluate R(state, prm). */
  virtual void res(const HostVector& state,
                   const HostVector& prm,
                   HostVector&       out) const = 0;

  /** @brief Assemble or update Jacobians of R at (state, prm). */
  virtual void linearize(const HostVector& state,
                         const HostVector& prm,
                         Linearization&    out) const = 0;
};

} // namespace state
} // namespace femx
