#pragma once

#include <refem/fe/FiniteElement.hpp>

namespace refem
{

class LagrangeQuadQ1 : public FiniteElement
{
public:
  static constexpr index_type dim    = 2;
  static constexpr index_type nnodes = 4;
  static constexpr index_type ndofs  = 4;
  static constexpr index_type degree = 1;

  std::string name() const override
  {
    return "LagrangeQuadQ1";
  }

  index_type dim() const override
  {
    return dim;
  }

  index_type numNodes() const override
  {
    return nnodes;
  }

  index_type numDofsPerElement() const override
  {
    return ndofs;
  }

  index_type order() const override
  {
    return degree;
  }

  ReferenceElement referenceElement() const override
  {
    return ReferenceElement::Quadrilateral;
  }

  void calcShape(const QuadraturePoint& qp,
                 VectorView<real_type>  N) const override
  {
    const real_type r = qp[0];
    const real_type s = qp[1];

    N[0] = 0.25 * (1.0 - r) * (1.0 - s);
    N[1] = 0.25 * (1.0 + r) * (1.0 - s);
    N[2] = 0.25 * (1.0 + r) * (1.0 + s);
    N[3] = 0.25 * (1.0 - r) * (1.0 + s);
  }

  void calcShapeGrad(const QuadraturePoint& qp,
                     MatrixView<real_type>  dNdr) const override
  {
    const real_type r = qp[0];
    const real_type s = qp[1];

    dNdr(0, 0) = -0.25 * (1.0 - s);
    dNdr(0, 1) = -0.25 * (1.0 - r);

    dNdr(1, 0) = 0.25 * (1.0 - s);
    dNdr(1, 1) = -0.25 * (1.0 + r);

    dNdr(2, 0) = 0.25 * (1.0 + s);
    dNdr(2, 1) = 0.25 * (1.0 + r);

    dNdr(3, 0) = -0.25 * (1.0 + s);
    dNdr(3, 1) = 0.25 * (1.0 - r);
  }
};

} // namespace refem
