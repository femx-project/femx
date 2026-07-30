#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>

namespace femx
{
namespace fem
{

/**
 * @brief Own element-local to global degree-of-freedom mappings.
 *
 * Element degree-of-freedom counts may vary. A zero return from
 * numDofsPerElem() indicates that no single count applies to every element.
 */
class DofMap
{
public:
  DofMap() = default;

  /**
   * @brief Create a uniform element degree-of-freedom map.
   *
   * @param[in] num_dofs - Number of global degrees of freedom.
   * @param[in] num_elems - Number of elements.
   * @param[in] num_dofs_per_elem - Number of degrees of freedom per element.
   * @throws std::runtime_error - If a dimension is invalid.
   */
  DofMap(Index num_dofs, Index num_elems, Index num_dofs_per_elem);

  /**
   * @brief Create a possibly variable-size element degree-of-freedom map.
   *
   * @param[in] num_dofs - Number of global degrees of freedom.
   * @param[in] elem_offsets - Offsets into `elem_dofs`, starting at zero.
   * @param[in] elem_dofs - Flattened global degrees of freedom.
   * @throws std::runtime_error - If the offsets or degrees of freedom are
   * invalid.
   */
  DofMap(Index             num_dofs,
         HostVector<Index> elem_offsets,
         HostVector<Index> elem_dofs);

  /**
   * @brief Allocate and reset a uniform element degree-of-freedom map.
   *
   * @param[in] num_dofs - Number of global degrees of freedom.
   * @param[in] num_elems - Number of elements.
   * @param[in] num_dofs_per_elem - Number of degrees of freedom per element.
   * @throws std::runtime_error - If a dimension is invalid.
   */
  void allocate(Index num_dofs,
                Index num_elems,
                Index num_dofs_per_elem);

  /** @brief Return the number of elements. */
  Index numElems() const noexcept;

  /** @brief Return the number of global degrees of freedom. */
  Index numDofs() const noexcept;

  /** @brief Return the uniform element degree-of-freedom count, or zero if it varies. */
  Index numDofsPerElem() const noexcept;

  /**
   * @brief Return the number of degrees of freedom on one element.
   *
   * @param[in] ie - Element index.
   * @return Number of degrees of freedom on the element.
   * @throws std::runtime_error - If the element index is out of range.
   */
  Index numElementDofs(Index ie) const;

  /**
   * @brief Return the global degree of freedom for a local element index.
   *
   * @param[in] ie - Element index.
   * @param[in] il - Local degree-of-freedom index.
   * @return Global degree-of-freedom index.
   * @throws std::runtime_error - If an index is out of range.
   */
  Index elementDof(Index ie, Index il) const;

  /**
   * @brief Assign a global degree of freedom to a local element index.
   *
   * @param[in] ie - Element index.
   * @param[in] il - Local degree-of-freedom index.
   * @param[in] global_dof - Global degree-of-freedom index.
   * @throws std::runtime_error - If an index is out of range.
   */
  void setElementDof(Index ie, Index il, Index global_dof);

  /**
   * @brief Return the contiguous global degrees of freedom for one element.
   *
   * @param[in] ie - Element index.
   * @return Non-owning view of the element's global degrees of freedom.
   * @throws std::runtime_error - If the element index is out of range.
   */
  HostVectorView<const Index> elementDofs(Index ie) const;

private:
  void validate();

  Index offset(Index ie, Index il) const;

  Index             num_dofs_{0};          ///< Number of global degrees of freedom.
  Index             num_dofs_per_elem_{0}; ///< Uniform element size, or zero.
  HostVector<Index> elem_offsets_;         ///< Offsets into flattened element data.
  HostVector<Index> elem_dofs_;            ///< Flattened global degrees of freedom.
};

} // namespace fem
} // namespace femx
