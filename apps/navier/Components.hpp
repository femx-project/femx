#pragma once

#include <femx/common/Types.hpp>

namespace femx::navier
{

constexpr Index kMaxLocalNodes            = 16;
constexpr Index kMaxLocalDofs             = 64;
constexpr Index kMaxLocalQuadraturePoints = 64;
constexpr Index kMaxSpatialDim            = 3;

struct KernelFluid
{
  Real rho = 1.0;
  Real mu  = 1.0;
};

struct LocalElementValues
{
  Index       nq   = 0;
  Index       nn   = 0;
  Index       dim  = 0;
  const Real* N    = nullptr;
  const Real* dNdx = nullptr;
  const Real* JxW  = nullptr;

  Real shape(Index qp, Index node) const
  {
    return N[qp * nn + node];
  }

  Real grad(Index qp, Index node, Index comp) const
  {
    return dNdx[(qp * nn + node) * dim + comp];
  }

  Real wt(Index qp) const
  {
    return JxW[qp];
  }
};

struct LocalMatrix
{
  Index size = 0;
  Real* vals = nullptr;

  Real& operator()(Index row, Index col) const
  {
    return vals[row * size + col];
  }
};

struct LocalVector
{
  Index size = 0;
  Real* vals = nullptr;

  Real& operator[](Index row) const
  {
    return vals[row];
  }
};

struct QPState
{
  Real u[3]{};
  Real u_adv[3]{};
  Real grad_u[3][3]{};
  Real u_adv_grad_u[3]{};
  Real tau[3]{};
};

Index vdof(Index node, Index comp, Index dim);
Index pdof(Index node, Index nn, Index dim);
Index numLocalDofs(Index nn, Index dim);

void zeroLocalSystem(Index nd, LocalMatrix Ke, LocalVector Fe);

void updateQpStates(const LocalElementValues& ev,
                    const KernelFluid&        fluid,
                    Real                      dt,
                    const Real*               state,
                    const Real*               adv_state,
                    QPState*                  qps);

void assembleMassLHS(const LocalElementValues& ev,
                     const KernelFluid&        fluid,
                     Real                      dt,
                     LocalMatrix               Ke);

void assembleMassRHS(const LocalElementValues& ev,
                     const QPState*            qps,
                     const KernelFluid&        fluid,
                     Real                      dt,
                     LocalVector               Fe);

void assembleAdvectionLHS(const LocalElementValues& ev,
                          const QPState*            qps,
                          const KernelFluid&        fluid,
                          LocalMatrix               Ke);

void assembleAdvectionRHS(const LocalElementValues& ev,
                          const QPState*            qps,
                          const KernelFluid&        fluid,
                          LocalVector               Fe);

void assembleDiffusionLHS(const LocalElementValues& ev,
                          const KernelFluid&        fluid,
                          LocalMatrix               Ke);

void assembleDiffusionRHS(const LocalElementValues& ev,
                          const QPState*            qps,
                          const KernelFluid&        fluid,
                          LocalVector               Fe);

void assemblePreVelCouplingLHS(const LocalElementValues& ev,
                               LocalMatrix               Ke);

void assembleStabilizationLHS(const LocalElementValues& ev,
                              const QPState*            qps,
                              const KernelFluid&        fluid,
                              Real                      dt,
                              LocalMatrix               Ke);

void assembleStabilizationRHS(const LocalElementValues& ev,
                              const QPState*            qps,
                              const KernelFluid&        fluid,
                              Real                      dt,
                              LocalVector               Fe);

void finishLocalResidual(Index       nd,
                         const Real* nxt,
                         LocalMatrix Ke,
                         LocalVector Fe,
                         Real*       out);

} // namespace femx::navier
