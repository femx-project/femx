#pragma once

#include <femx/common/Types.hpp>

namespace femx
{
namespace fem
{

class FESpace;
class MixedFESpace;

/**
 * @brief Non-owning view of elem-to-global id connectivity.
 *
 * DofLayout provides a uniform element-dof traversal interface for scalar and
 * mixed finite-element spaces.
 */
class DofLayout
{
public:
  /** @brief View the element DOF layout of a scalar finite-element space. */
  explicit DofLayout(const FESpace& space);

  /** @brief View the concatenated element DOF layout of a mixed space. */
  explicit DofLayout(const MixedFESpace& space);

  /** @brief Return the number of elements. */
  Index numElems() const;

  /** @brief Return the number of global DOFs. */
  Index numDofs() const;

  /**
   * @brief Return a fixed local DOF count, or zero for variable-size layouts.
   */
  Index numDofsPerElem() const;

  /** @brief Replace `dofs` with the global DOFs of element `ie`. */
  void elemDofs(Index ie, Array<Index>& dofs) const;

private:
  const MixedFESpace& mixedSpace() const;

private:
  const FESpace*      fe_space_{nullptr};
  const MixedFESpace* mixed_space_{nullptr};
};

} // namespace fem
} // namespace femx
