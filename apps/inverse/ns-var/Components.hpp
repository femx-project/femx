#pragma once

#include <femx/core/Types.hpp>

namespace femx
{

struct FluidParams
{
  Real rho = 1.0;
  Real mu  = 1.0;
};

struct Qp
{
  Real u[3]{};
  Real grad[3][3]{};
  Real adv_grad[3]{};
  Real tau{0.0};
};

void evalQp(Index       num_shape,
            Index       dim,
            const Real* N,
            const Real* dNdx,
            const Real* x,
            Qp&         qp);

Real elemLength(Index num_shape, Index dim, const Real* dNdx, const Real u[3]);

Real glsTau(const Real u[3], Real rho, Real mu, Real dt, Real h, Index dim);

void updateElemState(Qp*         qps,
                     Index       num_qp,
                     Index       num_shape,
                     Index       dim,
                     const Real* N,
                     const Real* dNdx,
                     const Real* x,
                     Real        rho,
                     Real        mu,
                     Real        dt);

void assembleMassLHS(Index       num_shape,
                     Index       dim,
                     Real        rho,
                     Real        dt,
                     const Real* N,
                     Real        Jw,
                     Real*       Ke);

void assembleAdvectionLHS(Index       num_shape,
                          Index       dim,
                          Real        rho,
                          const Real* N,
                          const Real* dNdx,
                          Real        Jw,
                          const Qp&   qp,
                          Real*       Ke);

void assembleDiffusionLHS(Index       num_shape,
                          Index       dim,
                          Real        mu,
                          const Real* dNdx,
                          Real        Jw,
                          Real*       Ke);

void assemblePressureLHS(Index       num_shape,
                         Index       dim,
                         const Real* N,
                         const Real* dNdx,
                         Real        Jw,
                         Real*       Ke);

void assembleStabilizationLHS(Index       num_shape,
                              Index       dim,
                              Real        rho,
                              Real        dt,
                              const Real* N,
                              const Real* dNdx,
                              Real        Jw,
                              const Qp&   qp,
                              Real*       Ke);

void assembleMassRHS(Index       num_shape,
                     Index       dim,
                     Real        rho,
                     Real        dt,
                     const Real* N,
                     Real        Jw,
                     const Qp&   qp,
                     Real*       Fe);

void assembleAdvectionRHS(Index       num_shape,
                          Index       dim,
                          Real        rho,
                          const Real* N,
                          Real        Jw,
                          const Qp&   qp,
                          Real*       Fe);

void assembleDiffusionRHS(Index       num_shape,
                          Index       dim,
                          Real        mu,
                          const Real* dNdx,
                          Real        Jw,
                          const Qp&   qp,
                          Real*       Fe);

void assembleStabilizationRHS(Index       num_shape,
                              Index       dim,
                              Real        rho,
                              Real        dt,
                              const Real* dNdx,
                              Real        Jw,
                              const Qp&   qp,
                              Real*       Fe);

} // namespace femx
