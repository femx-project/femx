#pragma once

#include <femx/common/Types.hpp>

namespace femx::model::ns
{

/** @brief Store incompressible-fluid material properties. */
struct FluidProperties
{
  Real rho = 1.0; ///< Fluid density.
  Real mu  = 1.0; ///< Dynamic viscosity.
};

} // namespace femx::model::ns
