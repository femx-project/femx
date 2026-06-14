#pragma once

#include <vector>

#include <femx/core/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/CsrPattern.hpp>

namespace femx
{
namespace assembly
{

class SparsityPatternBuilder
{
public:
  static CsrPattern build(const FESpace& space)
  {
    return CsrPattern(space.numDofs(), space.numDofs(), collect(space));
  }

  static CsrPattern build(const MixedFESpace& space)
  {
    return CsrPattern(space.numDofs(), space.numDofs(), collect(space));
  }

private:
  static std::vector<std::vector<index_type>> collect(const FESpace& space)
  {
    std::vector<std::vector<index_type>> cdofs(
        static_cast<std::size_t>(space.numElems()));
    for (index_type cell = 0; cell < space.numElems(); ++cell)
    {
      cdofs[static_cast<std::size_t>(cell)] = space.elemDofs(cell);
    }
    return cdofs;
  }

  static std::vector<std::vector<index_type>> collect(
      const MixedFESpace& space)
  {
    std::vector<std::vector<index_type>> cdofs(
        static_cast<std::size_t>(space.numElems()));
    for (index_type cell = 0; cell < space.numElems(); ++cell)
    {
      cdofs[static_cast<std::size_t>(cell)] = space.elemDofs(cell);
    }
    return cdofs;
  }
};

} // namespace assembly
} // namespace femx
