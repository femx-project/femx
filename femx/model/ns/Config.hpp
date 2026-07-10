#pragma once

#include <string>

#include <femx/common/Types.hpp>

namespace femx::model::ns
{

struct FluidParams
{
  Real rho = 1.0; ///< Fluid density.
  Real mu  = 1.0; ///< Dynamic viscosity.
};

struct BoundarySelector
{
  Index       physical = 0; ///< Physical boundary tag.
  std::string name;         ///< Physical boundary name.
};

} // namespace femx::model::ns
