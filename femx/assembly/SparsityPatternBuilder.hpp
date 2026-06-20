#pragma once

#include <femx/core/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/algebra/CsrPattern.hpp>
#include <femx/algebra/IndexSetList.hpp>
#include <femx/algebra/Vector.hpp>

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
