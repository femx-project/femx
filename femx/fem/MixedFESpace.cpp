#include <stdexcept>

#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>

namespace femx
{
namespace fem
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

Index MixedFieldView::localDof(Index shape_idx,
                               Index comp) const noexcept
{
  return local_offset_ + space_->localDof(shape_idx, comp);
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
    throw std::runtime_error("MixedFESpace: no fields");
  }

  local_offsets_.resize(numFields());
  global_offsets_.resize(numFields());
  Index num_dofs_per_elem = 0;
  Index num_dofs          = 0;

  const Mesh* mesh = &fields_[0].mesh();
  for (Index fid = 0; fid < numFields(); ++fid)
  {
    FESpace& field = fields_[fid];
    if (&field.mesh() != mesh)
    {
      throw std::runtime_error("MixedFESpace: fields must share a mesh");
    }

    field.setup();
    local_offsets_[fid]   = num_dofs_per_elem;
    global_offsets_[fid]  = num_dofs;
    num_dofs_per_elem    += field.numDofsPerElem();
    num_dofs             += field.numDofs();
  }

  dof_map_.allocate(num_dofs,
                    fields_[0].numElems(),
                    num_dofs_per_elem);
  for (Index ie = 0; ie < dof_map_.numElems(); ++ie)
  {
    Index local_offset = 0;
    for (Index fid = 0; fid < numFields(); ++fid)
    {
      const FESpace&              field      = fields_[fid];
      HostVectorView<const Index> field_dofs = field.dofMap().elementDofs(ie);
      for (Index il = 0; il < field_dofs.size(); ++il)
      {
        dof_map_.setElementDof(
            ie,
            local_offset + il,
            global_offsets_[fid] + field_dofs[il]);
      }
      local_offset += field_dofs.size();
    }
  }
}

MixedFieldView MixedFESpace::field(Index fid) const
{
  if (fid < 0 || fid >= fields_.size())
  {
    throw std::runtime_error("MixedFESpace: field identifier is out of range");
  }

  return MixedFieldView(&fields_[fid], local_offsets_[fid], global_offsets_[fid]);
}

const Mesh& MixedFESpace::mesh() const noexcept
{
  return fields_[0].mesh();
}

const DofMap& MixedFESpace::dofMap() const noexcept
{
  return dof_map_;
}

Index MixedFESpace::numFields() const noexcept
{
  return fields_.size();
}

Index MixedFESpace::numElems() const noexcept
{
  return fields_.empty() ? 0 : fields_[0].numElems();
}

Index MixedFESpace::numDofs() const noexcept
{
  return dof_map_.numDofs();
}

Index MixedFESpace::numDofsPerElem() const noexcept
{
  return dof_map_.numDofsPerElem();
}

void MixedFESpace::elemDofs(Index              ie,
                            HostVector<Index>& dofs) const
{
  dofs = dof_map_.elementDofs(ie);
}

HostVector<Index> MixedFESpace::elemDofs(Index ie) const
{
  return HostVector<Index>(dof_map_.elementDofs(ie));
}

} // namespace fem
} // namespace femx
