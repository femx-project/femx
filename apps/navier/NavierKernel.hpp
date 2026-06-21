#pragma once

#include <femx/linalg/Vector.hpp>
#include <femx/assembly/EnzymeTimeVolumeKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>

namespace femx::navier
{

void NavierResidual(Index       step,
                    Index       cell,
                    Index       num_qp,
                    Index       num_nodes,
                    Index       dim,
                    Index       num_res,
                    Index       num_prev_states,
                    Index       num_next_states,
                    Index       num_prm,
                    const Real* N,
                    const Real* dNdx,
                    const Real* JxW,
                    const Real* prev_state,
                    const Real* next_state,
                    const Real* prm,
                    Real*       out);

using NavierKernel =
    assembly::EnzymeTimeVolumeKernel<NavierResidual>;

Vector<Real> physicalParams(Real rho, Real mu, Real dt);

NavierKernel makeNavierKernel(const FESpace&         velocity_space,
                              const GaussQuadrature& quadrature,
                              Index                  local_size,
                              Real                   rho,
                              Real                   mu,
                              Real                   dt);

} // namespace femx::navier
