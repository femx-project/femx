#include "Components.hpp"

#include <algorithm>
#include <stdexcept>

#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>

namespace femx
{
namespace model
{
namespace ns
{

HostNavierData makeNavierData(const fem::FESpace&         vel_sp,
                              const fem::GaussQuadrature& quad)
{
  const Index num_elems = vel_sp.mesh().numElems();
  const Index num_qpts  = quad.size();
  const Index num_nodes = vel_sp.numShapesPerElem();
  const Index dim       = vel_sp.numComponents();
  const Index num_dofs  = (dim + 1) * num_nodes;
  if (num_elems <= 0 || num_qpts <= 0 || num_nodes <= 0 || dim <= 0
      || dim > kMaxDim || num_qpts > kMaxNq || num_nodes > kMaxNn
      || num_dofs > kMaxNd)
  {
    throw std::runtime_error("NavierData received unsupported dimensions");
  }

  fem::ElementValues vals(vel_sp.finiteElement(), quad);
  if (vals.numQuadraturePoints() != num_qpts
      || vals.numNodes() != num_nodes || vals.dim() != dim)
  {
    throw std::runtime_error("NavierData element dimensions are inconsistent");
  }

  HostNavierData data;
  data.num_elems_ = num_elems;
  data.num_qpts_  = num_qpts;
  data.num_nodes_ = num_nodes;
  data.dim_       = dim;
  data.N_.resize(num_qpts * num_nodes);
  data.dNdx_.resize(num_elems * num_qpts * num_nodes * dim);
  data.JxW_.resize(num_elems * num_qpts);

  std::copy(vals.NData(),
            vals.NData() + data.N_.size(),
            data.N_.begin());
  const Index grad_size = num_qpts * num_nodes * dim;
  for (Index ie = 0; ie < num_elems; ++ie)
  {
    vals.reinit(vel_sp.mesh().elem(ie));
    std::copy(vals.dNdxData(),
              vals.dNdxData() + grad_size,
              data.dNdx_.begin() + ie * grad_size);
    std::copy(vals.JxWData(),
              vals.JxWData() + num_qpts,
              data.JxW_.begin() + ie * num_qpts);
  }
  return data;
}

} // namespace ns
} // namespace model
} // namespace femx
