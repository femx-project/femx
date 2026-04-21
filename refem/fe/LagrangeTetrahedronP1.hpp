#pragma once

#include <refem/fe/FiniteElement.hpp>

namespace refem
{

class LagrangeTetrahedronP1 : public FiniteElement
{
public:
  static constexpr index_type dim = 3;
  static constexpr index_type nnodes = 4;
  static constexpr index_type ndofs  = 4;
  static constexpr index_type degree = 1;

  std::string name() const override
  {
    return "LagrangeTetrahedronP1";
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
    return ReferenceElement::Tetrahedron;
  }

  void calcShape(const QuadraturePoint& qp,
                 VectorView<real_type>  N) const override
  {
    const real_type r = qp[0];
    const real_type s = qp[1];
    const real_type t = qp[2];

    N[0] = 1.0 - r - s - t;
    N[1] = r;
    N[2] = s;
    N[3] = t;
  }

  void calcShapeGrad(const QuadraturePoint&,
                     MatrixView<real_type> dNdr) const override
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

} // namespace refem
