#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace equation
{

/** @brief Nonlinear state equation represented by a residual R(u, m) = 0. */
class ResidualEquation
{
public:
  virtual ~ResidualEquation() = default;

  virtual index_type numStates() const    = 0;
  virtual index_type numParams() const    = 0;
  virtual index_type numResiduals() const = 0;

  virtual void residual(const Vector& state,
                        const Vector& params,
                        Vector&       out) const = 0;

  virtual void applyStateJac(const Vector& state,
                             const Vector& params,
                             const Vector& dir,
                             Vector&       out) const = 0;

  virtual void applyStateJacT(const Vector& state,
                              const Vector& params,
                              const Vector& lambda,
                              Vector&       out) const = 0;

  virtual void applyParamJac(const Vector& state,
                             const Vector& params,
                             const Vector& dir,
                             Vector&       out) const = 0;

  virtual void applyParamJacT(const Vector& state,
                              const Vector& params,
                              const Vector& lambda,
                              Vector&       out) const = 0;
};

} // namespace equation
} // namespace femx
