#include <stdexcept>

#include <refem/fe/FESpace.hpp>

namespace refem
{

FESpace::FESpace(const Mesh*          mesh,
                 const FiniteElement* finite_element,
                 index_type           components)
  : mesh_(mesh),
    finite_element_(finite_element),
    components_(components)
{
  if (!mesh_ || !finite_element_)
  {
    throw std::runtime_error("FESpace: null mesh or finite element");
  }
  if (components_ <= 0)
  {
    throw std::runtime_error("FESpace: invalid component count");
  }
}

void FESpace::setup()
{
  const index_type num_elem = mesh_->numElems();
  num_shapes_per_elem_      = finite_element_->numDofsPerElement();
  const index_type ndof_e   = components_ * num_shapes_per_elem_;

  dof_map_.allocate(num_elem, ndof_e);
  num_dofs_ = components_ * mesh_->numNodes();

  for (index_type e = 0; e < num_elem; ++e)
  {
    const index_type* conn = mesh_->cellNodeIds(e);
    for (index_type a = 0; a < num_shapes_per_elem_; ++a)
    {
      for (index_type c = 0; c < components_; ++c)
      {
        dof_map_.setElementDof(e, localDof(a, c), globalDof(conn[a], c));
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
  return *finite_element_;
}

const DofMap& FESpace::dofMap() const noexcept
{
  return dof_map_;
}

index_type FESpace::numElems() const noexcept
{
  return mesh_->numElems();
}

index_type FESpace::numDofs() const noexcept
{
  return num_dofs_;
}

index_type FESpace::numComponents() const noexcept
{
  return components_;
}

index_type FESpace::numShapesPerElem() const noexcept
{
  return num_shapes_per_elem_;
}

index_type FESpace::numDofsPerElem() const noexcept
{
  return dof_map_.numElementDofs();
}

index_type FESpace::localDof(index_type shape_index,
                             index_type component) const noexcept
{
  return components_ * shape_index + component;
}

index_type FESpace::globalDof(index_type node,
                              index_type component) const noexcept
{
  return components_ * node + component;
}

std::vector<index_type> FESpace::elemDofs(index_type cell) const
{
  std::vector<index_type> dofs(
      static_cast<std::size_t>(dof_map_.numElementDofs()));

  const index_type* data = dof_map_.elementDofsData(cell);
  for (index_type i = 0; i < dof_map_.numElementDofs(); ++i)
  {
    dofs[static_cast<std::size_t>(i)] = data[i];
  }

  return dofs;
}

} // namespace refem
