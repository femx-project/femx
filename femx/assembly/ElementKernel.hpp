#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace assembly
{

/** @brief Cell-local residual and Jacobian kernel. */
class ElementKernel
{
public:
  virtual ~ElementKernel() = default;

  virtual void res(index_type    cell,
                   const Vector& u,
                   const Vector& m,
                   Vector&       out) const = 0;

  virtual void stateJac(index_type    cell,
                        const Vector& u,
                        const Vector& m,
                        DenseMatrix&  out) const = 0;

  virtual void paramJac(index_type    cell,
                        const Vector& u,
                        const Vector& m,
                        DenseMatrix&  out) const = 0;
};

} // namespace assembly
} // namespace femx
