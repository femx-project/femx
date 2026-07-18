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

/**
 * @brief Compile ordered app boundary settings into unique constrained DOFs.
 *
 * Later entries replace earlier values at shared boundary nodes. This lets a
 * wall condition define inlet/outlet rims and cavity corners explicitly.
 */
fem::DirichletBC makeDirichletBC(
    const fem::MixedFESpace& space,
    const Array<BCsParams>&  bcs,
    Real                     time);

} // namespace femx::model::ns
