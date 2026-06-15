#include <stdexcept>

#include <femx/fem/FESpace.hpp>

namespace femx
{

FESpace::FESpace(const Mesh*          mesh,
                 const FiniteElement* finite_element,
                 Index                components)
  : mesh_(mesh),
    fe_(finite_element),
    components_(components)
{
  if (!mesh_ || !fe_)
  {
    throw std::runtime_error("FESpace: null mesh or finite elem");
  }
  if (components_ <= 0)
  {
    throw std::runtime_error("FESpace: invalid component count");
  }
}

void FESpace::setup()
{
  const Index num_elem = mesh_->numElems();
  num_shapes_per_elem_ = fe_->numDofsPerElement();
  const Index ndof_e   = components_ * num_shapes_per_elem_;

  dof_map_.allocate(num_elem, ndof_e);
  num_dofs_ = components_ * mesh_->numNodes();

  for (Index ic = 0; ic < num_elem; ++ic)
  {
    const auto& cell = mesh_->cell(ic);
    if (cell.numNodes() != fe_->numNodes())
    {
      throw std::runtime_error(
          "FESpace: finite elem node count does not match mesh cell");
    }

    const Index* conn = mesh_->cellNodeIds(ic);
    for (Index a = 0; a < num_shapes_per_elem_; ++a)
    {
      for (Index c = 0; c < components_; ++c)
      {
        dof_map_.setElementDof(ic, localDof(a, c), globalDof(conn[a], c));
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
  return components_;
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
                        Index component) const noexcept
{
  return components_ * shape_index + component;
}

Index FESpace::globalDof(Index in,
                         Index component) const noexcept
{
  return components_ * in + component;
}

void FESpace::elemDofs(Index               ic,
                       std::vector<Index>& dofs) const
{
  dofs.resize(static_cast<std::size_t>(dof_map_.numElementDofs()));

  const Index* data = dof_map_.elementDofsData(ic);
  for (Index i = 0; i < dof_map_.numElementDofs(); ++i)
  {
    dofs[static_cast<std::size_t>(i)] = data[i];
  }
}

std::vector<Index> FESpace::elemDofs(Index ic) const
{
  std::vector<Index> dofs;
  elemDofs(ic, dofs);
  return dofs;
}

} // namespace femx
