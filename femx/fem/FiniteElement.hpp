#pragma once

#include <string>

#include <femx/core/Types.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/ReferenceElement.hpp>
#include <femx/algebra/MatrixView.hpp>
#include <femx/algebra/VectorView.hpp>

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
  virtual Index dim() const = 0;

  /** @brief Return the number of interpolation nodes in one elem. */
  virtual Index numNodes() const = 0;

  /** @brief Return the number of scalar shape functions in one elem. */
  virtual Index numDofsPerElement() const = 0;

  /** @brief Return the polynomial order of the elem. */
  virtual Index order() const = 0;

  /** @brief Return the reference cell shape used by this elem. */
  virtual ReferenceElement referenceElement() const = 0;

  /** @brief Evaluate shape functions at a reference quad point. */
  virtual void calcShape(const QuadraturePoint& qp,
                         VectorView<Real>       N) const = 0;

  /** @brief Evaluate shape function gradients in reference coordinates. */
  virtual void calcShapeGrad(const QuadraturePoint& qp,
                             MatrixView<Real>       dNdxi) const = 0;
};

} // namespace femx
