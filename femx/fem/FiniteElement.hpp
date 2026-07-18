#pragma once

#include <string>

#include <femx/common/Types.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/ReferenceElement.hpp>
#include <femx/linalg/MatrixView.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{
namespace fem
{

/**
 * @brief Interface for interpolation on a reference element.
 *
 * Concrete finite elements provide shape-function values and reference
 * gradients at quadrature points.  The physical mapping is handled separately
 * by ElementValues.
 */
class FiniteElement
{
public:
  virtual ~FiniteElement() = default;

  /** @brief Return a human-readable elem type name. */
  virtual std::string name() const = 0;

  /** @brief Return the spatial dimension of the reference elem. */
  virtual Index dim() const = 0;

  /** @brief Return the number of interpolation nodes in one elem. */
  virtual Index numNodes() const = 0;

  /** @brief Return the number of scalar shape functions in one elem. */
  virtual Index numDofsPerElement() const = 0;

  /** @brief Return the polynomial order of the elem. */
  virtual Index order() const = 0;

  /** @brief Return the reference elem shape used by this elem. */
  virtual ReferenceElement referenceElement() const = 0;

  /**
   * @brief Evaluate shape functions at a reference quadrature point.
   *
   * @param[in] qp - Reference quadrature point.
   * @param[out] N - Output vector of length numDofsPerElement().
   */
  virtual void calcN(const QuadraturePoint& qp,
                     HostVectorView         N) const = 0;

  /**
   * @brief Evaluate shape-function gradients in reference coordinates.
   *
   * @param[in] qp - Reference quadrature point.
   * @param[out] dNdxi - Matrix with one row per shape and one column per
   * reference coordinate.
   */
  virtual void calcdNdr(const QuadraturePoint& qp,
                        MatrixView<Real>       dNdxi) const = 0;
};

} // namespace fem
} // namespace femx
