#pragma once

#include <string>

#include <femx/common/Types.hpp>

namespace femx::navier
{

struct FluidParams
{
  Real rho = 1.0;
  Real mu  = 1.0;
};

struct BoundarySelector
{
  Index       physical = 0;
  std::string name;
};

} // namespace femx::navier
