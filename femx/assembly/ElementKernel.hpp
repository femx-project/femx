#pragma once

#include <femx/core/Types.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace assembly
{

/** @brief Cell-local residual and Jacobian kernel. */
class ElementKernel
{
public:
  virtual ~ElementKernel() = default;

  virtual void res(Index               ic,
                   const Vector<Real>& u,
                   const Vector<Real>& m,
                   Vector<Real>&       out) const = 0;

  virtual void stateJac(Index               ic,
                        const Vector<Real>& u,
                        const Vector<Real>& m,
                        DenseMatrix&        out) const = 0;

  virtual void paramJac(Index               ic,
                        const Vector<Real>& u,
                        const Vector<Real>& m,
                        DenseMatrix&        out) const = 0;
};

} // namespace assembly
} // namespace femx
