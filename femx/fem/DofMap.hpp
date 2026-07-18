#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

/**
 * @brief Element-local to global degree-of-freedom map.
 */
class DofMap
{
public:
  DofMap() = default;

  /** @brief Create an id map with storage for all elem dofs. */
  DofMap(Index num_elems, Index num_dofs_per_elem);

  /** @brief Allocate and reset the elem-to-global-id table. */
  void allocate(Index num_elems, Index num_dofs_per_elem);

  // Accessors
  Index numElements() const noexcept;
  Index numElementDofs() const noexcept;

  /** @brief Return the global id assigned to one local elem id. */
  Index elementDof(Index ie, Index il) const noexcept;

  /** @brief Assign a global id to one local elem id. */
  void setElementDof(Index ie, Index il, Index gdof) noexcept;

  /** @brief Return the contiguous global id list for one elem. */
  const Index* elementDofsData(Index ie) const noexcept;

private:
  /** @brief Return the flat storage offset for an elem-local id pair. */
  Index offset(Index ie, Index il) const noexcept;

  Index        num_elems_         = 0;
  Index        num_dofs_per_elem_ = 0;
  Array<Index> elem_dofs_;
};

} // namespace fem
} // namespace femx
