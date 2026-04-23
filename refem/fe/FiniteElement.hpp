#pragma once

#include <string>

#include <refem/common/Types.hpp>
#include <refem/fe/GaussQuadrature.hpp>
#include <refem/fe/ReferenceElement.hpp>
#include <refem/linalg/MatrixView.hpp>
#include <refem/linalg/VectorView.hpp>

namespace refem
{

/** @brief Interface for a finite element on a reference cell. */
class FiniteElement
{
public:
  virtual ~FiniteElement() = default;

  /** @brief Return a human-readable element type name. */
  virtual std::string name() const = 0;

  /** @brief Return the spatial dimension of the reference cell. */
  virtual index_type dim() const               = 0;

  /** @brief Return the number of interpolation nodes in one element. */
  virtual index_type numNodes() const          = 0;

  /** @brief Return the number of scalar shape functions in one element. */
  virtual index_type numDofsPerElement() const = 0;

  /** @brief Return the polynomial order of the element. */
  virtual index_type order() const             = 0;

  /** @brief Return the reference cell shape used by this element. */
  virtual ReferenceElement referenceElement() const = 0;

  /** @brief Evaluate shape functions at a reference quadrature point. */
  virtual void calcShape(const QuadraturePoint& qp,
                         VectorView<real_type>  N) const = 0;

  /** @brief Evaluate shape function gradients in reference coordinates. */
  virtual void calcShapeGrad(const QuadraturePoint& qp,
                             MatrixView<real_type>  dNdxi) const = 0;
};

} // namespace refem
