#pragma once

#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{

class DofMap
{
public:
  DofMap() = default;
  /** @brief Create a dof map with storage for all elem dofs. */
  DofMap(Index num_elems, Index num_dofs_per_elem);

  /** @brief Allocate and reset the elem-to-global-dof table. */
  void allocate(Index num_elems, Index num_dofs_per_elem);

  // Accessors
  Index numElements() const noexcept;
  Index numElementDofs() const noexcept;

  /** @brief Return the global dof assigned to one local elem dof. */
  Index elementDof(Index ie, Index il) const noexcept;

  /** @brief Assign a global dof to one local elem dof. */
  void setElementDof(Index ie, Index il, Index gdof) noexcept;

  /** @brief Return the contiguous global dof list for one elem. */
  const Index* elementDofsData(Index ie) const noexcept;

private:
  /** @brief Return the flat storage offset for an elem-local dof pair. */
  Index offset(Index ie, Index il) const noexcept;

  Index         num_elems_         = 0;
  Index         num_dofs_per_elem_ = 0;
  Vector<Index> elem_dofs_;
};

} // namespace femx
