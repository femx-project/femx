#include <stdexcept>

#include <femx/fem/FESpace.hpp>

namespace femx
{
namespace fem
{

FESpace::FESpace(const Mesh*          mesh,
                 const FiniteElement* finite_element,
                 Index                comps)
  : mesh_(mesh),
    fe_(finite_element),
    comps_(comps)
{
  if (!mesh_ || !fe_)
  {
    throw std::runtime_error("FESpace: null mesh or finite elem");
  }
  if (comps_ <= 0)
  {
    throw std::runtime_error("FESpace: invalid component count");
  }
}

void FESpace::setup()
{
  const Index num_elem      = mesh_->numElems();
  num_shapes_per_elem_      = fe_->numDofsPerElement();
  const Index num_elem_dofs = comps_ * num_shapes_per_elem_;

  dof_map_.allocate(num_elem, num_elem_dofs);
  num_dofs_ = comps_ * mesh_->numNodes();

  for (Index ie = 0; ie < num_elem; ++ie)
  {
    const auto& elem = mesh_->elem(ie);
    if (elem.numNodes() != fe_->numNodes())
    {
      throw std::runtime_error(
          "FESpace: finite elem node count does not match mesh elem");
    }

    const Index* conn = mesh_->elemNodeIds(ie);
    for (Index a = 0; a < num_shapes_per_elem_; ++a)
    {
      for (Index c = 0; c < comps_; ++c)
      {
        dof_map_.setElementDof(ie, localDof(a, c), globalDof(conn[a], c));
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
  return num_dofs_;
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
  return dof_map_.numElementDofs();
}

Index FESpace::localDof(Index shape_index,
                        Index comp) const noexcept
{
  return comps_ * shape_index + comp;
}

Index FESpace::globalDof(Index in,
                         Index comp) const noexcept
{
  return comps_ * in + comp;
}

void FESpace::elemDofs(Index         ie,
                       Array<Index>& dofs) const
{
  dofs.resize(dof_map_.numElementDofs());

  const Index* data = dof_map_.elementDofsData(ie);
  for (Index i = 0; i < dof_map_.numElementDofs(); ++i)
  {
    dofs[i] = data[i];
  }
}

Array<Index> FESpace::elemDofs(Index ie) const
{
  Array<Index> dofs;
  elemDofs(ie, dofs);
  return dofs;
}

} // namespace fem
} // namespace femx
