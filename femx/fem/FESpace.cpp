#include <stdexcept>

#include <femx/fem/FESpace.hpp>

using namespace std;

namespace femx
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
    throw runtime_error("FESpace: null mesh or finite elem");
  }
  if (comps_ <= 0)
  {
    throw runtime_error("FESpace: invalid component count");
  }
}

void FESpace::setup()
{
  const Index num_elem = mesh_->numElems();
  num_shapes_per_elem_ = fe_->numDofsPerElement();
  const Index ndof_e   = comps_ * num_shapes_per_elem_;

  dof_map_.allocate(num_elem, ndof_e);
  nd_ = comps_ * mesh_->numNodes();

  for (Index ic = 0; ic < num_elem; ++ic)
  {
    const auto& cell = mesh_->cell(ic);
    if (cell.numNodes() != fe_->numNodes())
    {
      throw runtime_error(
          "FESpace: finite elem node count does not match mesh cell");
    }

    const Index* conn = mesh_->cellNodeIds(ic);
    for (Index a = 0; a < num_shapes_per_elem_; ++a)
    {
      for (Index c = 0; c < comps_; ++c)
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
  return nd_;
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

void FESpace::elemDofs(Index          ic,
                       Vector<Index>& dofs) const
{
  dofs.resize(dof_map_.numElementDofs());

  const Index* data = dof_map_.elementDofsData(ic);
  for (Index i = 0; i < dof_map_.numElementDofs(); ++i)
  {
    dofs[i] = data[i];
  }
}

Vector<Index> FESpace::elemDofs(Index ic) const
{
  Vector<Index> dofs;
  elemDofs(ic, dofs);
  return dofs;
}

} // namespace femx
