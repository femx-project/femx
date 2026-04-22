#pragma once

#include <vector>

#include <refem/common/Types.hpp>

namespace refem
{

class DofMap
{
public:
  DofMap() = default;
  DofMap(index_type num_elements, index_type num_dofs_per_elem);

  void allocate(index_type num_elements, index_type num_dofs_per_elem);

  index_type        numElements() const noexcept;
  index_type        numElementDofs() const noexcept;
  index_type        elementDof(index_type elem_id, index_type local_id) const noexcept;
  void              setElementDof(index_type elem_id, index_type local_id, index_type gdof) noexcept;
  const index_type* elementDofsData(index_type elem_id) const noexcept;

private:
  std::size_t offset(index_type elem_id, index_type local_id) const noexcept;

  index_type              num_elements_      = 0;
  index_type              num_dofs_per_elem_ = 0;
  std::vector<index_type> elem_dofs_;
};

} // namespace refem
