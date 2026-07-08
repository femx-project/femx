#pragma once

#include "ForwardConfig.hpp"
#include <femx/fem/BoundaryCondition.hpp>

namespace femx
{
class MixedFESpace;
} // namespace femx

namespace femx::model::ns
{

DirichletCondition makeBoundaryCondition(
    const femx::MixedFESpace& space,
    const Vector<BCsParams>&  bcs,
    Real                      time);

} // namespace femx::model::ns
