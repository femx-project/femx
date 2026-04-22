#include <stdexcept>

#include <refem/fe/DofMap.hpp>

namespace refem
{

DofMap::DofMap(index_type num_elements, index_type num_dofs_per_elem)
{
  allocate(num_elements, num_dofs_per_elem);
}

void DofMap::allocate(index_type num_elements, index_type num_dofs_per_elem)
{
  if (num_elements < 0 || num_dofs_per_elem <= 0)
  {
    throw std::runtime_error("DofMap: invalid size");
  }

  num_elements_      = num_elements;
  num_dofs_per_elem_ = num_dofs_per_elem;
  elem_dofs_.assign(static_cast<std::size_t>(num_elements_) * static_cast<std::size_t>(num_dofs_per_elem_), 0);
}

index_type DofMap::numElements() const noexcept
{
  return num_elements_;
}

index_type DofMap::numElementDofs() const noexcept
{
  return num_dofs_per_elem_;
}

index_type DofMap::elementDof(index_type elem_id, index_type local_id) const noexcept
{
  return elem_dofs_[offset(elem_id, local_id)];
}

void DofMap::setElementDof(index_type elem_id, index_type local_id, index_type gdof) noexcept
{
  elem_dofs_[offset(elem_id, local_id)] = gdof;
}

const index_type* DofMap::elementDofsData(index_type elem_id) const noexcept
{
  return elem_dofs_.data() + static_cast<std::size_t>(elem_id) * static_cast<std::size_t>(num_dofs_per_elem_);
}

std::size_t DofMap::offset(index_type elem_id, index_type local_id) const noexcept
{
  return static_cast<std::size_t>(elem_id) * static_cast<std::size_t>(num_dofs_per_elem_) + static_cast<std::size_t>(local_id);
}

} // namespace refem
