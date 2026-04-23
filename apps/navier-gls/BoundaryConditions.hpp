#pragma once

#include "Config.hpp"
#include <refem/bc/DirichletCondition.hpp>

namespace refem
{
class MixedFESpace;
}

namespace refem
{

DirichletCondition makeBoundaryCondition(
    const MixedFESpace&           space,
    const std::vector<BCsParams>& bcs,
    real_type                     time);

} // namespace refem
