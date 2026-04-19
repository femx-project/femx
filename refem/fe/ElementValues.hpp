#pragma once

#include <vector>

#include <refem/common/Types.hpp>
#include <refem/linalg/MatrixView.hpp>
#include <refem/linalg/VectorView.hpp>

namespace refem
{

class Cell;
class FiniteElement;
class GaussQuadrature;

class ElementValues
{
public:
  ElementValues(const FiniteElement&  finite_element,
                const GaussQuadrature& quadrature);

  void reinit(const Cell& cell);

  index_type numNodes() const;
  index_type numDofs() const;
  index_type dim() const;
  index_type numQuadraturePoints() const;

  VectorView<const real_type> N(index_type q) const;
  MatrixView<const real_type> dNdr(index_type q) const;
  MatrixView<const real_type> dNdx(index_type q) const;

  real_type detJ(index_type q) const;
  real_type weight(index_type q) const;
  real_type JxW(index_type q) const;

private:
  void calcReferenceValues();
  void calcPhysicalValues(const Cell& cell);

  static real_type invJacobian(const std::vector<real_type>& J,
                               std::vector<real_type>&       invJ,
                               index_type                    dim);

private:
  const FiniteElement*  finite_element_{nullptr};
  const GaussQuadrature* quadrature_{nullptr};

  index_type num_nodes_{0};
  index_type dim_{0};
  index_type num_qp_{0};

  std::vector<real_type> N_;
  std::vector<real_type> dNdr_;
  std::vector<real_type> dNdx_;

  std::vector<real_type> detJ_;
  std::vector<real_type> weights_;

  std::vector<real_type> J_;
  std::vector<real_type> invJ_;
};

} // namespace refem
