#pragma once

#include <string>

#include <femx/common/Types.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/ReferenceElement.hpp>
#include <femx/linalg/MatrixView.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{

/** @brief Interface for a finite elem on a reference cell. */
class FiniteElement
{
public:
  virtual ~FiniteElement() = default;

  /** @brief Return a human-readable elem type name. */
  virtual std::string name() const = 0;

  /** @brief Return the spatial dimension of the reference cell. */
  virtual index_type dim() const               = 0;

  /** @brief Return the number of interpolation nodes in one elem. */
  virtual index_type numNodes() const          = 0;

  /** @brief Return the number of scalar shape functions in one elem. */
  virtual index_type numDofsPerElement() const = 0;

  /** @brief Return the polynomial order of the elem. */
  virtual index_type order() const             = 0;

  /** @brief Return the reference cell shape used by this elem. */
  virtual ReferenceElement referenceElement() const = 0;

  /** @brief Evaluate shape functions at a reference quad point. */
  virtual void calcShape(const QuadraturePoint& qp,
                         VectorView<real_type>  N) const = 0;

  /** @brief Evaluate shape function gradients in reference coordinates. */
  virtual void calcShapeGrad(const QuadraturePoint& qp,
                             MatrixView<real_type>  dNdxi) const = 0;
};

} // namespace femx
