#pragma once

#include <femx/fem/FiniteElement.hpp>

namespace femx
{

class LagrangeTetrahedronP1 : public FiniteElement
{
public:
  static constexpr Index spatial_dim = 3;
  static constexpr Index nnodes      = 4;
  static constexpr Index nd          = 4;
  static constexpr Index degree      = 1;

  std::string name() const override
  {
    return "LagrangeTetrahedronP1";
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
    return ReferenceElement::Tetrahedron;
  }

  void calcShape(const QuadraturePoint& qp,
                 VectorView<Real>       N) const override
  {
    const Real r = qp[0];
    const Real s = qp[1];
    const Real t = qp[2];

    N[0] = 1.0 - r - s - t;
    N[1] = r;
    N[2] = s;
    N[3] = t;
  }

  void calcShapeGrad(const QuadraturePoint&,
                     MatrixView<Real> dNdr) const override
  {
    dNdr(0, 0) = -1.0;
    dNdr(0, 1) = -1.0;
    dNdr(0, 2) = -1.0;

    dNdr(1, 0) = 1.0;
    dNdr(1, 1) = 0.0;
    dNdr(1, 2) = 0.0;

    dNdr(2, 0) = 0.0;
    dNdr(2, 1) = 1.0;
    dNdr(2, 2) = 0.0;

    dNdr(3, 0) = 0.0;
    dNdr(3, 1) = 0.0;
    dNdr(3, 2) = 1.0;
  }
};

} // namespace femx
