#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Reduced objective F(m) seen by optimizers. */
class ReducedFunctional
{
public:
  virtual ~ReducedFunctional() = default;

  virtual index_type numParams() const = 0;

  virtual real_type value(const Vector& params) = 0;

  virtual void grad(const Vector& params, Vector& out) = 0;

  virtual real_type valueGrad(const Vector& params, Vector& grad_out)
  {
    const real_type obj_val = value(params);
    grad(params, grad_out);
    return obj_val;
  }
};

} // namespace inverse
} // namespace femx
