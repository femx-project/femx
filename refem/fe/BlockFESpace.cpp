#include <stdexcept>

#include <refem/fe/BlockFESpace.hpp>
#include <refem/mesh/Mesh.hpp>

namespace refem
{

BlockFieldView::BlockFieldView(const FESpace* space,
                               index_type     local_offset,
                               index_type     global_offset)
  : space_(space),
    local_offset_(local_offset),
    global_offset_(global_offset)
{
}

const FESpace& BlockFieldView::space() const noexcept
{
  return *space_;
}

index_type BlockFieldView::numComponents() const noexcept
{
  return space_->numComponents();
}

index_type BlockFieldView::numShapesPerElem() const noexcept
{
  return space_->numShapesPerElem();
}

index_type BlockFieldView::numDofsPerElem() const noexcept
{
  return space_->numDofsPerElem();
}

index_type BlockFieldView::localDof(index_type shape_index,
                                    index_type component) const noexcept
{
  return local_offset_ + space_->localDof(shape_index, component);
}

index_type BlockFieldView::globalDof(index_type scalar_dof,
                                     index_type component) const noexcept
{
  return global_offset_ + space_->globalDof(scalar_dof, component);
}

void BlockFESpace::addField(const FESpace& space)
{
  fields_.push_back(space);
}

void BlockFESpace::setup()
{
  if (fields_.empty())
  {
    throw std::runtime_error("BlockFESpace: no fields");
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
      throw std::runtime_error("BlockFESpace: fields must share a mesh");
    }

    field.setup();
    local_offsets_[i]   = num_dofs_per_elem_;
    global_offsets_[i]  = num_dofs_;
    num_dofs_per_elem_ += field.numDofsPerElem();
    num_dofs_          += field.numDofs();
  }
}

BlockFieldView BlockFESpace::field(index_type field_id) const
{
  if (field_id < 0 || field_id >= static_cast<index_type>(fields_.size()))
  {
    throw std::runtime_error("BlockFESpace: field id out of range");
  }

  const auto id = static_cast<std::size_t>(field_id);
  return BlockFieldView(&fields_[id], local_offsets_[id], global_offsets_[id]);
}

const Mesh& BlockFESpace::mesh() const noexcept
{
  return fields_[0].mesh();
}

index_type BlockFESpace::numFields() const noexcept
{
  return static_cast<index_type>(fields_.size());
}

index_type BlockFESpace::numElems() const noexcept
{
  return fields_[0].numElems();
}

index_type BlockFESpace::numDofs() const noexcept
{
  return num_dofs_;
}

index_type BlockFESpace::numDofsPerElem() const noexcept
{
  return num_dofs_per_elem_;
}

std::vector<index_type> BlockFESpace::elemDofs(index_type cell) const
{
  std::vector<index_type> dofs;
  dofs.reserve(static_cast<std::size_t>(num_dofs_per_elem_));

  for (std::size_t i = 0; i < fields_.size(); ++i)
  {
    const auto field_dofs = fields_[i].elemDofs(cell);
    for (index_type dof : field_dofs)
    {
      dofs.push_back(global_offsets_[i] + dof);
    }
  }

  return dofs;
}

} // namespace refem
