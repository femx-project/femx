#include <stdexcept>

#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>

using namespace std;

namespace femx
{

MixedFieldView::MixedFieldView(const FESpace* space,
                               Index          local_offset,
                               Index          global_offset)
  : space_(space),
    local_offset_(local_offset),
    global_offset_(global_offset)
{
}

const FESpace& MixedFieldView::space() const noexcept
{
  return *space_;
}

Index MixedFieldView::numComponents() const noexcept
{
  return space_->numComponents();
}

Index MixedFieldView::numShapesPerElem() const noexcept
{
  return space_->numShapesPerElem();
}

Index MixedFieldView::numDofsPerElem() const noexcept
{
  return space_->numDofsPerElem();
}

Index MixedFieldView::localDof(Index shape_index,
                               Index comp) const noexcept
{
  return local_offset_ + space_->localDof(shape_index, comp);
}

Index MixedFieldView::globalDof(Index scalar_dof,
                                Index comp) const noexcept
{
  return global_offset_ + space_->globalDof(scalar_dof, comp);
}

void MixedFESpace::addField(const FESpace& space)
{
  fields_.push_back(space);
}

void MixedFESpace::setup()
{
  if (fields_.empty())
  {
    throw runtime_error("MixedFESpace: no fields");
  }

  local_offsets_.resize(numFields());
  global_offsets_.resize(numFields());
  ndpe_ = 0;
  nd_   = 0;

  const Mesh* mesh = &fields_[0].mesh();
  for (Index fid = 0; fid < numFields(); ++fid)
  {
    FESpace& field = fields_[fid];
    if (&field.mesh() != mesh)
    {
      throw runtime_error("MixedFESpace: fields must share a mesh");
    }

    field.setup();
    local_offsets_[fid]   = ndpe_;
    global_offsets_[fid]  = nd_;
    ndpe_                += field.numDofsPerElem();
    nd_                  += field.numDofs();
  }
}

MixedFieldView MixedFESpace::field(Index fid) const
{
  if (fid < 0 || fid >= fields_.size())
  {
    throw runtime_error("MixedFESpace: field id out of range");
  }

  return MixedFieldView(&fields_[fid], local_offsets_[fid], global_offsets_[fid]);
}

const Mesh& MixedFESpace::mesh() const noexcept
{
  return fields_[0].mesh();
}

Index MixedFESpace::numFields() const noexcept
{
  return fields_.size();
}

Index MixedFESpace::numElems() const noexcept
{
  return fields_[0].numElems();
}

Index MixedFESpace::numDofs() const noexcept
{
  return nd_;
}

Index MixedFESpace::numDofsPerElem() const noexcept
{
  return ndpe_;
}

void MixedFESpace::elemDofs(Index          ie,
                            Vector<Index>& dofs) const
{
  dofs.resize(ndpe_);

  Index offset = 0;
  for (Index fid = 0; fid < numFields(); ++fid)
  {
    const FESpace& field      = fields_[fid];
    const Index*   field_dofs = field.dofMap().elementDofsData(ie);
    for (Index j = 0; j < field.numDofsPerElem(); ++j)
    {
      dofs[offset + j] = global_offsets_[fid] + field_dofs[j];
    }
    offset += field.numDofsPerElem();
  }
}

Vector<Index> MixedFESpace::elemDofs(Index ie) const
{
  Vector<Index> dofs;
  elemDofs(ie, dofs);
  return dofs;
}

} // namespace femx
