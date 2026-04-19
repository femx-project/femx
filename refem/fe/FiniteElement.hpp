#pragma once

#include <string>

#include <refem/common/Types.hpp>
#include <refem/fe/GaussQuadrature.hpp>
#include <refem/fe/ReferenceElement.hpp>
#include <refem/linalg/MatrixView.hpp>
#include <refem/linalg/VectorView.hpp>

namespace refem
{

class FiniteElement
{
public:
  virtual ~FiniteElement() = default;

  virtual std::string name() const = 0;

  virtual index_type dim() const               = 0;
  virtual index_type numNodes() const          = 0;
  virtual index_type numDofsPerElement() const = 0;
  virtual index_type order() const             = 0;

  virtual ReferenceElement referenceElement() const = 0;

  virtual void calcShape(const QuadraturePoint& qp,
                         VectorView<real_type>  N) const = 0;

  virtual void calcShapeGrad(const QuadraturePoint& qp,
                             MatrixView<real_type>  dNdxi) const = 0;
};

} // namespace refem
