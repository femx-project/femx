#pragma once

#include "ForwardConfig.hpp"
#include <femx/fem/DirichletBC.hpp>

namespace femx
{
namespace fem
{
class MixedFESpace;
} // namespace fem
} // namespace femx

namespace femx::model::ns
{

fem::DirichletBC makeDirichletBC(
    const fem::MixedFESpace& space,
    const Array<BCsParams>&  bcs,
    Real                     time);

} // namespace femx::model::ns
