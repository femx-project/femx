#pragma once

#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>
#include <femx/problem/Linearization.hpp>

namespace femx
{
namespace problem
{

struct Dimensions
{
  Index num_states    = 0;
  Index num_params    = 0;
  Index num_residuals = 0;
};

class Residual
{
public:
  virtual ~Residual() = default;

  virtual Dimensions dimensions() const = 0;

  virtual void residual(const Vector<Real>& state,
                        const Vector<Real>& prm,
                        Vector<Real>&       out) const = 0;

  virtual void linearize(const Vector<Real>& state,
                         const Vector<Real>& prm,
                         Linearization&      out) const = 0;
};

} // namespace problem
} // namespace femx
