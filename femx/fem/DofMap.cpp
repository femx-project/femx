#include <stdexcept>

#include <femx/fem/DofMap.hpp>

namespace femx
{

DofMap::DofMap(Index num_elems, Index num_dofs_per_elem)
{
  allocate(num_elems, num_dofs_per_elem);
}

void DofMap::allocate(Index num_elems, Index num_dofs_per_elem)
{
  if (num_elems < 0 || num_dofs_per_elem <= 0)
  {
    throw std::runtime_error("DofMap: invalid size");
  }

  num_elems_         = num_elems;
  num_dofs_per_elem_ = num_dofs_per_elem;
  elem_dofs_.resize(num_elems_ * num_dofs_per_elem_);
}

Index DofMap::numElements() const noexcept
{
  return num_elems_;
}

Index DofMap::numElementDofs() const noexcept
{
  return num_dofs_per_elem_;
}

Index DofMap::elementDof(Index ie, Index il) const noexcept
{
  return elem_dofs_[offset(ie, il)];
}

void DofMap::setElementDof(Index ie, Index il, Index gdof) noexcept
{
  elem_dofs_[offset(ie, il)] = gdof;
}

const Index* DofMap::elementDofsData(Index ie) const noexcept
{
  return elem_dofs_.data() + ie * num_dofs_per_elem_;
}

Index DofMap::offset(Index ie, Index il) const noexcept
{
  return ie * num_dofs_per_elem_ + il;
}

} // namespace femx
