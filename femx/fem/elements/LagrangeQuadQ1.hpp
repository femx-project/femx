#pragma once

#include <femx/fem/FiniteElement.hpp>

namespace femx
{

class LagrangeQuadQ1 : public FiniteElement
{
public:
  static constexpr Index spatial_dim = 2;
  static constexpr Index nnodes      = 4;
  static constexpr Index nd          = 4;
  static constexpr Index degree      = 1;

  std::string name() const override
  {
    return "LagrangeQuadQ1";
  }

  Index dim() const override
  {
    return spatial_dim;
  }

  Index numNodes() const override
  {
    return nnodes;
  }

  Index numDofsPerElement() const override
  {
    return nd;
  }

  Index order() const override
  {
    return degree;
  }

  ReferenceElement referenceElement() const override
  {
    return ReferenceElement::Quadrilateral;
  }

  void calcShape(const QuadraturePoint& qp,
                 VectorView<Real>       N) const override
  {
    const Real r = qp[0];
    const Real s = qp[1];

    N[0] = 0.25 * (1.0 - r) * (1.0 - s);
    N[1] = 0.25 * (1.0 + r) * (1.0 - s);
    N[2] = 0.25 * (1.0 + r) * (1.0 + s);
    N[3] = 0.25 * (1.0 - r) * (1.0 + s);
  }

  void calcShapeGrad(const QuadraturePoint& qp,
                     MatrixView<Real>       dNdr) const override
  {
    const Real r = qp[0];
    const Real s = qp[1];

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

} // namespace femx
