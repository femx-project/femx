#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/common/View.hpp>

namespace femx
{
namespace fem
{

class FiniteElement;
class GaussQuadrature;
class Mesh;

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

  /**
   * @brief Recompute physical values for one mesh element.
   *
   * @param[in] mesh - Mesh containing the element.
   * @param[in] ie - Element index.
   */
  void reinit(const Mesh& mesh, Index ie);

  Index numNodes() const;
  Index numDofs() const;
  Index dim() const;
  Index numQuadraturePoints() const;

  HostVectorView<const Real> N(Index iq) const;
  HostMatrixView<const Real> dNdr(Index iq) const;
  HostMatrixView<const Real> dNdx(Index iq) const;

  Real detJ(Index iq) const;
  Real wt(Index iq) const;
  Real JxW(Index iq) const;

  const Real* NData() const;
  const Real* dNdxData() const;
  const Real* JxWData() const;

private:
  void calcReferenceValues();
  void calcPhysicalValues(const Mesh& mesh, Index ie);

  static Real invJacobian(const HostVector<Real>& J,
                          HostVector<Real>&       invJ,
                          Index                   dim);

private:
  const FiniteElement*   fe_{nullptr};
  const GaussQuadrature* quad_{nullptr};

  Index num_nodes_{0};
  Index num_dofs_{0};
  Index dim_{0};
  Index num_qpts_{0};

  HostVector<Real> N_;    ///< Shape values at quadrature points.
  HostVector<Real> dNdr_; ///< Reference gradients at quadrature points.
  HostVector<Real> dNdx_; ///< Physical gradients at quadrature points.

  HostVector<Real> detJ_; ///< Jacobian determinants.
  HostVector<Real> wts_;  ///< Quadrature weights.
  HostVector<Real> JxW_;  ///< Weighted Jacobian determinants.

  HostVector<Real> J_;    ///< Element Jacobian workspace.
  HostVector<Real> invJ_; ///< Inverse Jacobian workspace.
};

} // namespace fem
} // namespace femx
