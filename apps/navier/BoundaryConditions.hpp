#pragma once

#include "Config.hpp"
#include <femx/fem/DirichletBC.hpp>

namespace femx
{
namespace fem
{
class MixedFESpace;
} // namespace fem
} // namespace femx

namespace femx::apps::navier
{

/**
 * @brief Compile ordered app boundary settings into unique constrained DOFs.
 *
 * Later entries replace earlier values at shared boundary nodes. This lets a
 * wall condition define inlet/outlet rims and cavity corners explicitly.
 *
 * @param[in] space - Mixed velocity-pressure finite-element space.
 * @param[in] bcs   - Ordered boundary configurations.
 * @param[in] time  - Evaluation time.
 * @return Unique Dirichlet degrees of freedom and values.
 */
fem::DirichletBC makeDirichletBoundaryConditions(
    const fem::MixedFESpace&                   space,
    const HostVector<BoundaryConditionConfig>& bcs,
    Real                                       time);

} // namespace femx::apps::navier
