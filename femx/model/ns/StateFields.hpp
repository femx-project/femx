#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/View.hpp>

namespace femx::model::ns
{

/**
 * @brief Extract nodal velocity components and pressure from a state vector.
 *
 * @param[in] state - Mixed Navier-Stokes state vector.
 * @param[in] space - Mixed velocity-pressure finite-element space.
 * @param[out] ux - Nodal x-velocity values.
 * @param[out] uy - Nodal y-velocity values.
 * @param[out] uz - Nodal z-velocity values.
 * @param[out] pressure - Nodal pressure values.
 * @throws std::runtime_error - If the input or output dimensions are invalid.
 */
void splitStateFields(HostConstVectorView      state,
                      const fem::MixedFESpace& space,
                      HostVector&              ux,
                      HostVector&              uy,
                      HostVector&              uz,
                      HostVector&              pressure);

} // namespace femx::model::ns
