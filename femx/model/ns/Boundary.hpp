#pragma once

#include "ForwardConfig.hpp"
#include <femx/fem/BoundaryCondition.hpp>

namespace femx
{
namespace fem
{
class MixedFESpace;
} // namespace fem
} // namespace femx

namespace femx::model::ns
{

fem::DirichletCondition makeBoundaryCondition(
    const fem::MixedFESpace& space,
    const Vector<BCsParams>& bcs,
    Real                     time);

} // namespace femx::model::ns
