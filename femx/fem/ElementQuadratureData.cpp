#include <algorithm>
#include <stdexcept>

#include <femx/fem/ElementQuadratureData.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>

namespace femx
{
namespace fem
{

HostElementQuadratureData makeElementQuadratureData(
    const FESpace&         space,
    const GaussQuadrature& quad)
{
  const FiniteElement& finite_element = space.finiteElement();
  if (space.numShapesPerElem() != finite_element.numDofsPerElement())
  {
    throw std::runtime_error(
        "Element quadrature data requires an initialized finite-element space");
  }
  if (finite_element.referenceElement() != quad.referenceElement())
  {
    throw std::runtime_error(
        "Finite element and quadrature reference elements do not match");
  }

  ElementValues vals(finite_element, quad);

  HostElementQuadratureData data;
  data.num_elems_  = space.numElems();
  data.num_qpts_   = vals.numQuadraturePoints();
  data.num_shapes_ = vals.numDofs();
  data.dim_        = vals.dim();

  data.N_.resize(data.num_qpts_ * data.num_shapes_);
  data.dNdx_.resize(
      data.num_elems_ * data.num_qpts_ * data.num_shapes_ * data.dim_);
  data.JxW_.resize(data.num_elems_ * data.num_qpts_);

  std::copy(vals.NData(), vals.NData() + data.N_.size(), data.N_.begin());

  const Index gradient_size =
      data.num_qpts_ * data.num_shapes_ * data.dim_;
  for (Index ie = 0; ie < data.num_elems_; ++ie)
  {
    vals.reinit(space.mesh().elem(ie));
    std::copy(vals.dNdxData(),
              vals.dNdxData() + gradient_size,
              data.dNdx_.begin() + ie * gradient_size);
    std::copy(vals.JxWData(),
              vals.JxWData() + data.num_qpts_,
              data.JxW_.begin() + ie * data.num_qpts_);
  }

  return data;
}

} // namespace fem
} // namespace femx
