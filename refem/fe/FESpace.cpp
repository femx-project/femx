#include <stdexcept>

#include <refem/fe/FESpace.hpp>

namespace refem
{

FESpace::FESpace(const Mesh* mesh, const FiniteElement* finite_element)
  : mesh_(mesh),
    finite_element_(finite_element)
{
  if (!mesh_ || !finite_element_)
  {
    throw std::runtime_error("FESpace: null mesh or finite element");
  }
}

void FESpace::setup()
{
  const index_type num_elem = mesh_->numCells();
  const index_type ndof_e   = finite_element_->numDofsPerElement();

  dof_map_.allocate(num_elem, ndof_e);
  num_dofs_ = mesh_->numNodes();

  for (index_type e = 0; e < num_elem; ++e)
  {
    const index_type* conn = mesh_->cellNodeIds(e);
    for (index_type a = 0; a < ndof_e; ++a)
    {
      dof_map_.setElementDof(e, a, conn[a]);
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

index_type FESpace::numCells() const noexcept
{
  return mesh_->numCells();
}

index_type FESpace::numDofs() const noexcept
{
  return num_dofs_;
}

std::vector<index_type> FESpace::cellDofs(index_type cell) const
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
