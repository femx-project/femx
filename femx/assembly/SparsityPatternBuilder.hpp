#pragma once

#include <vector>

#include <femx/common/Types.hpp>
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
  static std::vector<std::vector<Index>> collect(const FESpace& space)
  {
    std::vector<std::vector<Index>> cdofs(
        static_cast<std::size_t>(space.numElems()));
    for (Index ic = 0; ic < space.numElems(); ++ic)
    {
      cdofs[static_cast<std::size_t>(ic)] = space.elemDofs(ic);
    }
    return cdofs;
  }

  static std::vector<std::vector<Index>> collect(
      const MixedFESpace& space)
  {
    std::vector<std::vector<Index>> cdofs(
        static_cast<std::size_t>(space.numElems()));
    for (Index ic = 0; ic < space.numElems(); ++ic)
    {
      cdofs[static_cast<std::size_t>(ic)] = space.elemDofs(ic);
    }
    return cdofs;
  }
};

} // namespace assembly
} // namespace femx
