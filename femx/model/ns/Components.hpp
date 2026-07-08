#pragma once

#include <femx/common/Types.hpp>

namespace femx::model::ns
{

constexpr Index kMaxNn  = 16;
constexpr Index kMaxNd  = 64;
constexpr Index kMaxNq  = 64;
constexpr Index kMaxDim = 3;

struct KernelFluid
{
  Real rho = 1.0;
  Real mu  = 1.0;
};

struct LocalElementValues
{
  Index       num_qpts  = 0;
  Index       num_nodes = 0;
  Index       dim       = 0;
  const Real* N_data    = nullptr;
  const Real* dNdx_data = nullptr;
  const Real* JxW_data  = nullptr;

  Real N(Index qp, Index in) const
  {
    return N_data[qp * num_nodes + in];
  }

  Real dNdx(Index qp, Index in, Index comp) const
  {
    return dNdx_data[(qp * num_nodes + in) * dim + comp];
  }

  Real JxW(Index qp) const
  {
    return JxW_data[qp];
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

Index vdof(Index in, Index comp, Index dim);
Index pdof(Index in, Index num_nodes, Index dim);
Index numLocalDofs(Index num_nodes, Index dim);

void zeroLocalSystem(Index num_dofs, LocalMatrix Ke, LocalVector Fe);

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

void finishLocalResidual(Index       num_dofs,
                         const Real* nxt,
                         LocalMatrix Ke,
                         LocalVector Fe,
                         Real*       out);

} // namespace femx::model::ns
