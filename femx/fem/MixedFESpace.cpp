#include <stdexcept>

#include <femx/fem/MixedFESpace.hpp>
#include <femx/mesh/Mesh.hpp>

namespace femx
{

MixedFieldView::MixedFieldView(const FESpace* space,
                               index_type     local_offset,
                               index_type     global_offset)
  : space_(space),
    local_offset_(local_offset),
    global_offset_(global_offset)
{
}

const FESpace& MixedFieldView::space() const noexcept
{
  return *space_;
}

index_type MixedFieldView::numComponents() const noexcept
{
  return space_->numComponents();
}

index_type MixedFieldView::numShapesPerElem() const noexcept
{
  return space_->numShapesPerElem();
}

index_type MixedFieldView::numDofsPerElem() const noexcept
{
  return space_->numDofsPerElem();
}

index_type MixedFieldView::localDof(index_type shape_index,
                                    index_type component) const noexcept
{
  return local_offset_ + space_->localDof(shape_index, component);
}

index_type MixedFieldView::globalDof(index_type scalar_dof,
                                     index_type component) const noexcept
{
  return global_offset_ + space_->globalDof(scalar_dof, component);
}

void MixedFESpace::addField(const FESpace& space)
{
  fields_.push_back(space);
}

void MixedFESpace::setup()
{
  if (fields_.empty())
  {
    throw std::runtime_error("MixedFESpace: no fields");
  }

  local_offsets_.assign(fields_.size(), 0);
  global_offsets_.assign(fields_.size(), 0);
  num_dofs_per_elem_ = 0;
  num_dofs_          = 0;

  const Mesh* mesh = &fields_[0].mesh();
  for (std::size_t i = 0; i < fields_.size(); ++i)
  {
    FESpace& field = fields_[i];
    if (&field.mesh() != mesh)
    {
      throw std::runtime_error("MixedFESpace: fields must share a mesh");
    }

    field.setup();
    local_offsets_[i]   = num_dofs_per_elem_;
    global_offsets_[i]  = num_dofs_;
    num_dofs_per_elem_ += field.numDofsPerElem();
    num_dofs_          += field.numDofs();
  }
}

MixedFieldView MixedFESpace::field(index_type field_id) const
{
  if (field_id < 0 || field_id >= static_cast<index_type>(fields_.size()))
  {
    throw std::runtime_error("MixedFESpace: field id out of range");
  }

  const auto id = static_cast<std::size_t>(field_id);
  return MixedFieldView(&fields_[id], local_offsets_[id], global_offsets_[id]);
}

const Mesh& MixedFESpace::mesh() const noexcept
{
  return fields_[0].mesh();
}

index_type MixedFESpace::numFields() const noexcept
{
  return static_cast<index_type>(fields_.size());
}

index_type MixedFESpace::numElems() const noexcept
{
  return fields_[0].numElems();
}

index_type MixedFESpace::numDofs() const noexcept
{
  return num_dofs_;
}

index_type MixedFESpace::numDofsPerElem() const noexcept
{
  return num_dofs_per_elem_;
}

void MixedFESpace::elemDofs(index_type               ic,
                            std::vector<index_type>& dofs) const
{
  dofs.resize(static_cast<std::size_t>(num_dofs_per_elem_));

  index_type offset = 0;
  for (std::size_t i = 0; i < fields_.size(); ++i)
  {
    const index_type* field_dofs = fields_[i].dofMap().elementDofsData(ic);
    for (index_type j = 0; j < fields_[i].numDofsPerElem(); ++j)
    {
      dofs[static_cast<std::size_t>(offset + j)] =
          global_offsets_[i] + field_dofs[j];
    }
    offset += fields_[i].numDofsPerElem();
  }
}

std::vector<index_type> MixedFESpace::elemDofs(index_type ic) const
{
  std::vector<index_type> dofs;
  elemDofs(ic, dofs);
  return dofs;
}

} // namespace femx
