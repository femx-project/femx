#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/Linearization.hpp>

namespace femx
{
namespace state
{

/**
 * @brief Sizes for a parameter-dependent residual problem.
 *
 * Dimensions records the state, parameter, and residual vector sizes expected
 * by residual evaluators and state solvers.
 */
struct Dimensions
{
  Index num_states    = 0;
  Index num_params    = 0;
  Index num_residuals = 0;
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
  virtual void res(const Vector<Real>& state,
                   const Vector<Real>& prm,
                   Vector<Real>&       out) const = 0;

  /** @brief Assemble or update Jacobians of R at (state, prm). */
  virtual void linearize(const Vector<Real>& state,
                         const Vector<Real>& prm,
                         Linearization&      out) const = 0;
};

} // namespace state
} // namespace femx
