#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/CsrPattern.hpp>

namespace femx
{
namespace assembly
{

class CsrBuilder
{
public:
  static CsrPattern build(const FESpace& space);

  static CsrPattern build(const MixedFESpace& space);

  static CsrPattern build(DofLayout layout);
};

} // namespace assembly
} // namespace femx
