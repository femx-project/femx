#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/Linearization.hpp>

namespace femx
{
namespace problem
{

struct Dimensions
{
  Index nst  = 0;
  Index nprm = 0;
  Index nres = 0;
};

class Residual
{
public:
  virtual ~Residual() = default;

  virtual Dimensions dims() const = 0;

  virtual void res(const Vector<Real>& state,
                   const Vector<Real>& prm,
                   Vector<Real>&       out) const = 0;

  virtual void linearize(const Vector<Real>& state,
                         const Vector<Real>& prm,
                         Linearization&      out) const = 0;
};

} // namespace problem
} // namespace femx
