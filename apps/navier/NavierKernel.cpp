#include "NavierKernel.hpp"

#include <stdexcept>

#include "Components.hpp"

namespace femx::navier
{
namespace
{

bool supportedLocalSize(Index num_qp,
                        Index num_nodes,
                        Index dim,
                        Index local_size)
{
  return dim > 0 && dim <= kMaxSpatialDim
         && num_nodes > 0 && num_nodes <= kMaxLocalNodes
         && num_qp > 0 && num_qp <= kMaxLocalQuadraturePoints
         && local_size > 0 && local_size <= kMaxLocalDofs
         && local_size == numLocalDofs(num_nodes, dim);
}

} // namespace

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
                    Real*       out)
{
  (void) step;
  (void) cell;

  for (Index i = 0; i < num_res; ++i)
  {
    out[i] = 0.0;
  }

  (void) num_prm;
  if (!supportedLocalSize(num_qp, num_nodes, dim, num_res)
      || num_prev_states != num_res || num_next_states != num_res)
  {
    return;
  }

  const Index       num_dofs = numLocalDofs(num_nodes, dim);
  const KernelFluid fluid{prm[0], prm[1]};
  const Real        dt = prm[2];

  LocalElementValues ev{num_qp, num_nodes, dim, N, dNdx, JxW};
  QPState            qps[kMaxLocalQuadraturePoints]{};
  Real               Ke_storage[kMaxLocalDofs * kMaxLocalDofs];
  Real               Fe_storage[kMaxLocalDofs];
  LocalMatrix        Ke{num_dofs, Ke_storage};
  LocalVector        Fe{num_dofs, Fe_storage};

  updateQpStates(ev, fluid, dt, prev_state, qps);
  zeroLocalSystem(num_dofs, Ke, Fe);

  assembleMassLHS(ev, fluid, dt, Ke);
  assembleAdvectionLHS(ev, qps, fluid, Ke);
  assembleDiffusionLHS(ev, fluid, Ke);
  assemblePreVelCouplingLHS(ev, Ke);
  assembleStabilizationLHS(ev, qps, fluid, dt, Ke);

  assembleMassRHS(ev, qps, fluid, dt, Fe);
  assembleAdvectionRHS(ev, qps, fluid, Fe);
  assembleDiffusionRHS(ev, qps, fluid, Fe);
  assembleStabilizationRHS(ev, qps, fluid, dt, Fe);

  finishLocalResidual(num_dofs, next_state, Ke, Fe, out);
}

Vector<Real> physicalParams(Real rho, Real mu, Real dt)
{
  if (rho <= 0.0 || mu < 0.0 || dt <= 0.0)
  {
    throw std::runtime_error("NavierKernel received invalid parameters");
  }

  Vector<Real> prm(3);
  prm[0] = rho;
  prm[1] = mu;
  prm[2] = dt;
  return prm;
}

NavierKernel makeNavierKernel(const FESpace&         velocity_space,
                              const GaussQuadrature& quadrature,
                              Index                  local_size,
                              Real                   rho,
                              Real                   mu,
                              Real                   dt)
{
  if (!supportedLocalSize(quadrature.size(),
                          velocity_space.numShapesPerElem(),
                          velocity_space.numComponents(),
                          local_size))
  {
    throw std::runtime_error("NavierKernel received unsupported element size");
  }

  return NavierKernel(velocity_space,
                      quadrature,
                      local_size,
                      local_size,
                      local_size,
                      physicalParams(rho, mu, dt));
}

} // namespace femx::navier
