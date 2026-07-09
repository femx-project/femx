#pragma once

#include <femx/common/Types.hpp>

namespace femx
{
template <typename T>
class Vector;

namespace inverse
{

/**
 * @brief Scalar objective functional J(u, m).
 *
 * Objective implementations provide value and first derivatives with respect
 * to state and parameter variables.  ReducedFunctional combines this interface
 * with a Residual to build parameter-only optimization problems.
 */
class Objective
{
public:
  virtual ~Objective() = default;

  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  /** @brief Evaluate J(state, prm). */
  virtual Real value(const Vector<Real>& state,
                     const Vector<Real>& prm) const = 0;

  /** @brief Compute dJ/du at (state, prm). */
  virtual void stateGrad(const Vector<Real>& state,
                         const Vector<Real>& prm,
                         Vector<Real>&       out) const = 0;

  /** @brief Compute dJ/dm at (state, prm). */
  virtual void paramGrad(const Vector<Real>& state,
                         const Vector<Real>& prm,
                         Vector<Real>&       out) const = 0;
};

} // namespace inverse
} // namespace femx
