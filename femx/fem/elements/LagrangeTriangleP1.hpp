#pragma once

#include <femx/fem/FiniteElement.hpp>

namespace femx
{

class LagrangeTriangleP1 : public FiniteElement
{
public:
  static constexpr Index spatial_dim = 2;
  static constexpr Index nnodes      = 3;
  static constexpr Index ndofs       = 3;
  static constexpr Index degree      = 1;

  std::string name() const override
  {
    return "LagrangeTriangleP1";
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
    return ndofs;
  }

  Index order() const override
  {
    return degree;
  }

  ReferenceElement referenceElement() const override
  {
    return ReferenceElement::Triangle;
  }

  void calcShape(const QuadraturePoint& qp,
                 VectorView<Real>       N) const override
  {
    const Real r = qp[0];
    const Real s = qp[1];

    N[0] = 1.0 - r - s;
    N[1] = r;
    N[2] = s;
  }

  void calcShapeGrad(const QuadraturePoint&,
                     MatrixView<Real> dNdr) const override
  {
    dNdr(0, 0) = -1.0;
    dNdr(0, 1) = -1.0;

    dNdr(1, 0) = 1.0;
    dNdr(1, 1) = 0.0;

    dNdr(2, 0) = 0.0;
    dNdr(2, 1) = 1.0;
  }
};

} // namespace femx
