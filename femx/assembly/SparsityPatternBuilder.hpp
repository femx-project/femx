#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/IndexSetList.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace assembly
{

class SparsityPatternBuilder
{
public:
  static CsrPattern build(const FESpace& space);

  static CsrPattern build(const MixedFESpace& space);

private:
  static IndexSetList collect(const FESpace& space);

  static IndexSetList collect(const MixedFESpace& space);
};

} // namespace assembly
} // namespace femx
