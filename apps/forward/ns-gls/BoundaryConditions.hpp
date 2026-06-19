#pragma once

#include "Config.hpp"
#include <femx/bc/DirichletCondition.hpp>

namespace femx
{
class MixedFESpace;
}

namespace femx
{

DirichletCondition makeBoundaryCondition(
    const MixedFESpace&           space,
    const std::vector<BCsParams>& bcs,
    Real                          time);

} // namespace femx
