#pragma once

#include <femx/assembly/BoundaryResidualEquation.hpp>
#include <femx/assembly/ElementResidualEquation.hpp>

namespace femx
{
namespace assembly
{

using FEMResidual         = ElementResidualEquation;
using BoundaryFEMResidual = BoundaryResidualEquation;

} // namespace assembly
} // namespace femx
