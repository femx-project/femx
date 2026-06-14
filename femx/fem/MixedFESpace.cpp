#include <stdexcept>

#include <femx/fem/MixedFESpace.hpp>
#include <femx/mesh/Mesh.hpp>

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
                               Index component) const noexcept
{
  return local_offset_ + space_->localDof(shape_index, component);
}

Index MixedFieldView::globalDof(Index scalar_dof,
                                Index component) const noexcept
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

MixedFieldView MixedFESpace::field(Index field_id) const
{
  if (field_id < 0 || field_id >= static_cast<Index>(fields_.size()))
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

Index MixedFESpace::numFields() const noexcept
{
  return static_cast<Index>(fields_.size());
}

Index MixedFESpace::numElems() const noexcept
{
  return fields_[0].numElems();
}

Index MixedFESpace::numDofs() const noexcept
{
  return num_dofs_;
}

Index MixedFESpace::numDofsPerElem() const noexcept
{
  return num_dofs_per_elem_;
}

void MixedFESpace::elemDofs(Index               ic,
                            std::vector<Index>& dofs) const
{
  dofs.resize(static_cast<std::size_t>(num_dofs_per_elem_));

  Index offset = 0;
  for (std::size_t i = 0; i < fields_.size(); ++i)
  {
    const Index* field_dofs = fields_[i].dofMap().elementDofsData(ic);
    for (Index j = 0; j < fields_[i].numDofsPerElem(); ++j)
    {
      dofs[static_cast<std::size_t>(offset + j)] =
          global_offsets_[i] + field_dofs[j];
    }
    offset += fields_[i].numDofsPerElem();
  }
}

std::vector<Index> MixedFESpace::elemDofs(Index ic) const
{
  std::vector<Index> dofs;
  elemDofs(ic, dofs);
  return dofs;
}

} // namespace femx
