#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/common/View.hpp>
#include <femx/fem/MixedFESpace.hpp>

namespace femx::model::navier
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
void splitStateFields(HostVectorView<const Real> state,
                      const fem::MixedFESpace&   space,
                      HostVector<Real>&          ux,
                      HostVector<Real>&          uy,
                      HostVector<Real>&          uz,
                      HostVector<Real>&          pressure);

} // namespace femx::model::navier
