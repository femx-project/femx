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
  elem_dofs_.assign(static_cast<std::size_t>(num_elems_) * static_cast<std::size_t>(num_dofs_per_elem_), 0);
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
  return elem_dofs_.data() + static_cast<std::size_t>(ie) * static_cast<std::size_t>(num_dofs_per_elem_);
}

std::size_t DofMap::offset(Index ie, Index il) const noexcept
{
  return static_cast<std::size_t>(ie) * static_cast<std::size_t>(num_dofs_per_elem_) + static_cast<std::size_t>(il);
}

} // namespace femx
