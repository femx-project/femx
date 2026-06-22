#pragma once

#include "Config.hpp"
#include <femx/fem/BoundaryCondition.hpp>

namespace femx
{
class MixedFESpace;
}

namespace femx
{

DirichletCondition makeBoundaryCondition(
    const MixedFESpace&      space,
    const Vector<BCsParams>& bcs,
    Real                     time);

} // namespace femx
