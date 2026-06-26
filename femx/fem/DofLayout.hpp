#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class FESpace;
class MixedFESpace;

/** @brief Non-owning view of elem-to-global id connectivity. */
class DofLayout
{
public:
  explicit DofLayout(const FESpace& space);

  explicit DofLayout(const MixedFESpace& space);

  Index numElems() const;

  Index numDofs() const;

  Index numDofsPerElem() const;

  void elemDofs(Index ie, Vector<Index>& dofs) const;

private:
  const MixedFESpace& mixedSpace() const;

private:
  const FESpace*      fe_space_{nullptr};
  const MixedFESpace* mixed_space_{nullptr};
};

} // namespace femx
