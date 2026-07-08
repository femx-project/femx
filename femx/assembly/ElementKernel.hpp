#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace assembly
{

/**
 * @brief Element-local residual and Jacobian kernel.
 *
 * Element kernels provide the physics-specific local contributions used by
 * FEMResidual and related residual wrappers.  Implementations receive the
 * global state and parameter vectors and write element-local outputs.
 */
class ElementKernel
{
public:
  virtual ~ElementKernel() = default;

  /**
   * @brief Compute the element residual contribution.
   *
   * @param[in] ie - Element index.
   * @param[in] u - Global state vector.
   * @param[in] m - Global parameter/control vector.
   * @param[out] out - Element-local residual values.
   */
  virtual void res(Index               ie,
                   const Vector<Real>& u,
                   const Vector<Real>& m,
                   Vector<Real>&       out) const = 0;

  /** @brief Compute the element Jacobian with respect to the state. */
  virtual void stateJac(Index               ie,
                        const Vector<Real>& u,
                        const Vector<Real>& m,
                        DenseMatrix&        out) const = 0;

  /** @brief Compute the element Jacobian with respect to the parameter. */
  virtual void paramJac(Index               ie,
                        const Vector<Real>& u,
                        const Vector<Real>& m,
                        DenseMatrix&        out) const = 0;
};

} // namespace assembly
} // namespace femx
