#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/MatrixView.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{
namespace fem
{

class Element;
class FiniteElement;
class GaussQuadrature;

/**
 * @brief Shape values, physical gradients, and weights for one element.
 *
 * ElementValues caches reference shape data and recomputes the physical
 * Jacobian-dependent quantities whenever reinit() is called for a new element.
 * Assembly kernels use this object to access N, dN/dx, detJ, and JxW at each
 * quadrature point.
 */
class ElementValues
{
public:
  /**
   * @brief Construct evaluator for one finite element and quadrature rule.
   *
   * @param[in] finite_element - Reference finite element.
   * @param[in] quad - Quadrature rule on the same reference element.
   */
  ElementValues(const FiniteElement&   finite_element,
                const GaussQuadrature& quad);

  /** @brief Recompute physical values for an element instance. */
  void reinit(const Element& elem);

  Index numNodes() const;
  Index numDofs() const;
  Index dim() const;
  Index numQuadraturePoints() const;

  HostConstVectorView    N(Index iq) const;
  MatrixView<const Real> dNdr(Index iq) const;
  MatrixView<const Real> dNdx(Index iq) const;

  Real detJ(Index iq) const;
  Real wt(Index iq) const;
  Real JxW(Index iq) const;

  const Real* NData() const;
  const Real* dNdxData() const;
  const Real* JxWData() const;

private:
  void calcReferenceValues();
  void calcPhysicalValues(const Element& elem);

  static Real invJacobian(const HostVector& J,
                          HostVector&       invJ,
                          Index             dim);

private:
  const FiniteElement*   fe_{nullptr};
  const GaussQuadrature* quad_{nullptr};

  Index num_nodes_{0};
  Index num_dofs_{0};
  Index dim_{0};
  Index num_qpts_{0};

  HostVector N_;    ///< Shape values at quadrature points.
  HostVector dNdr_; ///< Reference gradients at quadrature points.
  HostVector dNdx_; ///< Physical gradients at quadrature points.

  HostVector detJ_; ///< Jacobian determinants.
  HostVector wts_;  ///< Quadrature weights.
  HostVector JxW_;  ///< Weighted Jacobian determinants.

  HostVector J_;    ///< Element Jacobian workspace.
  HostVector invJ_; ///< Inverse Jacobian workspace.
};

} // namespace fem
} // namespace femx
