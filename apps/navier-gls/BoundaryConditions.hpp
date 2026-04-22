#pragma once

#include "Config.hpp"
#include <refem/bc/DirichletCondition.hpp>

namespace refem
{
class BlockFESpace;
}

namespace refem
{

DirichletCondition makeBoundaryCondition(
    const BlockFESpace&           space,
    const std::vector<BCsParams>& bcs,
    real_type                     time);

} // namespace refem
