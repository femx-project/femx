#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace assembly
{

/** @brief Elem-local residual and Jacobian kernel. */
class ElementKernel
{
public:
  virtual ~ElementKernel() = default;

  virtual void res(Index               ie,
                   const Vector<Real>& u,
                   const Vector<Real>& m,
                   Vector<Real>&       out) const = 0;

  virtual void stateJac(Index               ie,
                        const Vector<Real>& u,
                        const Vector<Real>& m,
                        DenseMatrix&        out) const = 0;

  virtual void paramJac(Index               ie,
                        const Vector<Real>& u,
                        const Vector<Real>& m,
                        DenseMatrix&        out) const = 0;
};

} // namespace assembly
} // namespace femx
