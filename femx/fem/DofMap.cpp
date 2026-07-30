#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/fem/DofMap.hpp>

namespace femx
{
namespace fem
{

DofMap::DofMap(Index num_dofs,
               Index num_elems,
               Index num_dofs_per_elem)
{
  allocate(num_dofs, num_elems, num_dofs_per_elem);
}

DofMap::DofMap(Index             num_dofs,
               HostVector<Index> elem_offsets,
               HostVector<Index> elem_dofs)
  : num_dofs_(num_dofs),
    elem_offsets_(std::move(elem_offsets)),
    elem_dofs_(std::move(elem_dofs))
{
  validate();
}

void DofMap::allocate(Index num_dofs,
                      Index num_elems,
                      Index num_dofs_per_elem)
{
  require(num_dofs >= 0 && num_elems >= 0 && num_dofs_per_elem > 0,
          "DofMap dimensions must be nonnegative with positive element size");
  const Index num_entries = num_elems * num_dofs_per_elem;

  num_dofs_          = num_dofs;
  num_dofs_per_elem_ = num_elems == 0 ? 0 : num_dofs_per_elem;
  elem_offsets_.resize(num_elems + 1);
  for (Index ie = 0; ie <= num_elems; ++ie)
  {
    elem_offsets_[ie] = ie * num_dofs_per_elem;
  }
  elem_dofs_.assign(num_entries, 0);
}

Index DofMap::numElems() const noexcept
{
  return elem_offsets_.empty() ? 0 : elem_offsets_.size() - 1;
}

Index DofMap::numDofs() const noexcept
{
  return num_dofs_;
}

Index DofMap::numDofsPerElem() const noexcept
{
  return num_dofs_per_elem_;
}

Index DofMap::numElementDofs(Index ie) const
{
  require(ie >= 0 && ie < numElems(),
          "DofMap element index is out of range");
  return elem_offsets_[ie + 1] - elem_offsets_[ie];
}

Index DofMap::elementDof(Index ie, Index il) const
{
  return elem_dofs_[offset(ie, il)];
}

void DofMap::setElementDof(Index ie, Index il, Index global_dof)
{
  require(global_dof >= 0 && global_dof < num_dofs_,
          "DofMap global degree of freedom is out of range");
  elem_dofs_[offset(ie, il)] = global_dof;
}

HostVectorView<const Index> DofMap::elementDofs(Index ie) const
{
  const Index count = numElementDofs(ie);
  return {elem_dofs_.data() + elem_offsets_[ie], count};
}

void DofMap::validate()
{
  require(num_dofs_ >= 0,
          "DofMap global DOF count must be nonnegative");
  require(!elem_offsets_.empty() && elem_offsets_[0] == 0,
          "DofMap element offsets must start at zero");
  require(elem_offsets_.back() == elem_dofs_.size(),
          "DofMap element offsets do not match flattened DOFs");

  num_dofs_per_elem_ = numElems() == 0
                           ? 0
                           : elem_offsets_[1] - elem_offsets_[0];
  for (Index ie = 0; ie < numElems(); ++ie)
  {
    require(elem_offsets_[ie + 1] > elem_offsets_[ie],
            "DofMap element offsets must increase");
    const Index count = elem_offsets_[ie + 1] - elem_offsets_[ie];
    if (count != num_dofs_per_elem_)
    {
      num_dofs_per_elem_ = 0;
    }
  }
  for (Index dof : elem_dofs_)
  {
    require(dof >= 0 && dof < num_dofs_,
            "DofMap global degree of freedom is out of range");
  }
}

Index DofMap::offset(Index ie, Index il) const
{
  const Index count = numElementDofs(ie);
  require(il >= 0 && il < count,
          "DofMap local degree of freedom is out of range");
  return elem_offsets_[ie] + il;
}

} // namespace fem
} // namespace femx
