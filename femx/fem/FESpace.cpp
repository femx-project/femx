#include <stdexcept>

#include <femx/fem/FESpace.hpp>

namespace femx
{
namespace fem
{

FESpace::FESpace(const Mesh*          mesh,
                 const FiniteElement* fe,
                 Index                comps)
  : mesh_(mesh),
    fe_(fe),
    comps_(comps)
{
  if (!mesh_ || !fe_)
  {
    throw std::runtime_error("FESpace: null mesh or finite element");
  }
  if (comps_ <= 0)
  {
    throw std::runtime_error("FESpace: invalid component count");
  }
  if (mesh_->dim() != fe_->dim())
  {
    throw std::runtime_error(
        "FESpace: mesh dimension does not match finite element");
  }
}

void FESpace::setup()
{
  const Index num_elem      = mesh_->numElems();
  num_shapes_per_elem_      = fe_->numDofsPerElement();
  const Index num_elem_dofs = comps_ * num_shapes_per_elem_;
  const Index num_dofs      = comps_ * mesh_->numNodes();

  dof_map_.allocate(num_dofs, num_elem, num_elem_dofs);

  for (Index ie = 0; ie < num_elem; ++ie)
  {
    if (mesh_->elemNumNodes(ie) != fe_->numNodes())
    {
      throw std::runtime_error(
          "FESpace: finite-element node count does not match mesh element");
    }
    if (mesh_->elemShape(ie) != ElementShape::Unknown
        && mesh_->elemShape(ie) != fe_->shape())
    {
      throw std::runtime_error(
          "FESpace: mesh element topology does not match finite element");
    }

    const auto conn = mesh_->elemNodeIds(ie);
    for (Index in = 0; in < num_shapes_per_elem_; ++in)
    {
      for (Index ic = 0; ic < comps_; ++ic)
      {
        dof_map_.setElementDof(
            ie, localDof(in, ic), globalDof(conn[in], ic));
      }
    }
  }
}

const Mesh& FESpace::mesh() const noexcept
{
  return *mesh_;
}

const FiniteElement& FESpace::finiteElement() const noexcept
{
  return *fe_;
}

const DofMap& FESpace::dofMap() const noexcept
{
  return dof_map_;
}

Index FESpace::numElems() const noexcept
{
  return mesh_->numElems();
}

Index FESpace::numDofs() const noexcept
{
  return dof_map_.numDofs();
}

Index FESpace::numComponents() const noexcept
{
  return comps_;
}

Index FESpace::numShapesPerElem() const noexcept
{
  return num_shapes_per_elem_;
}

Index FESpace::numDofsPerElem() const noexcept
{
  return dof_map_.numDofsPerElem();
}

Index FESpace::localDof(Index shape_idx,
                        Index comp) const noexcept
{
  return comps_ * shape_idx + comp;
}

Index FESpace::globalDof(Index in,
                         Index comp) const noexcept
{
  return comps_ * in + comp;
}

void FESpace::elemDofs(Index              ie,
                       HostVector<Index>& dofs) const
{
  dofs = dof_map_.elementDofs(ie);
}

HostVector<Index> FESpace::elemDofs(Index ie) const
{
  return HostVector<Index>(dof_map_.elementDofs(ie));
}

} // namespace fem
} // namespace femx
