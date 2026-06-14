#pragma once

#include <vector>

#include <femx/common/Types.hpp>

namespace femx
{

class DofMap
{
public:
  DofMap() = default;
  /** @brief Create a dof map with storage for all elem dofs. */
  DofMap(index_type num_elements, index_type num_dofs_per_elem);

  /** @brief Allocate and reset the elem-to-global-dof table. */
  void allocate(index_type num_elements, index_type num_dofs_per_elem);

  // Accessors
  index_type numElements() const noexcept;
  index_type numElementDofs() const noexcept;

  /** @brief Return the global dof assigned to one local elem dof. */
  index_type elementDof(index_type elem_id, index_type local_id) const noexcept;

  /** @brief Assign a global dof to one local elem dof. */
  void setElementDof(index_type elem_id, index_type local_id, index_type gdof) noexcept;

  /** @brief Return the contiguous global dof list for one elem. */
  const index_type* elementDofsData(index_type elem_id) const noexcept;

private:
  /** @brief Return the flat storage offset for an elem-local dof pair. */
  std::size_t offset(index_type elem_id, index_type local_id) const noexcept;

  index_type              num_elements_      = 0;
  index_type              num_dofs_per_elem_ = 0;
  std::vector<index_type> elem_dofs_;
};

} // namespace femx
