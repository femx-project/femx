#include "Components.hpp"

#include <cmath>

using namespace std;

namespace femx::navier
{
namespace
{

Real absoluteValue(Real value)
{
  return value >= 0.0 ? value : -value;
}

Real advectionDerivative(const LocalElementValues& ev,
                         const QPState&            qp,
                         Index                     iq,
                         Index                     node)
{
  Real value = 0.0;
  for (Index d = 0; d < ev.dim; ++d)
  {
    value += ev.grad(iq, node, d) * qp.u_adv[d];
  }
  return value;
}

Real shapeGradientDot(const LocalElementValues& ev,
                      Index                     iq,
                      Index                     i,
                      Index                     j)
{
  Real value = 0.0;
  for (Index d = 0; d < ev.dim; ++d)
  {
    value += ev.grad(iq, i, d) * ev.grad(iq, j, d);
  }
  return value;
}

void evalQp(const LocalElementValues& ev,
            Index                     iq,
            const Real*               state,
            const Real*               adv_state,
            QPState&                  qp)
{
  qp = {};
  for (Index node = 0; node < ev.nn; ++node)
  {
    for (Index c = 0; c < ev.dim; ++c)
    {
      const Real value  = state[vdof(node, c, ev.dim)];
      qp.u[c]          += ev.shape(iq, node) * value;
      for (Index d = 0; d < ev.dim; ++d)
      {
        qp.grad_u[c][d] += ev.grad(iq, node, d) * value;
      }
    }
  }

  for (Index node = 0; node < ev.nn; ++node)
  {
    for (Index d = 0; d < ev.dim; ++d)
    {
      qp.u_adv[d] +=
          ev.shape(iq, node) * adv_state[vdof(node, d, ev.dim)];
    }
  }

  for (Index c = 0; c < ev.dim; ++c)
  {
    for (Index d = 0; d < ev.dim; ++d)
    {
      qp.u_adv_grad_u[c] += qp.grad_u[c][d] * qp.u_adv[d];
    }
  }
}

Real elemLength(const LocalElementValues& ev, Index iq, const Real u[3])
{
  Real speed2 = 0.0;
  for (Index d = 0; d < ev.dim; ++d)
  {
    speed2 += u[d] * u[d];
  }

  const Real speed = sqrt(speed2);
  Real       dir[3]{};
  if (speed > 1.0e-10)
  {
    for (Index d = 0; d < ev.dim; ++d)
    {
      dir[d] = u[d] / speed;
    }
  }
  else
  {
    const Real value = 1.0 / sqrt(static_cast<Real>(ev.dim));
    for (Index d = 0; d < ev.dim; ++d)
    {
      dir[d] = value;
    }
  }

  Real sum = 0.0;
  for (Index node = 0; node < ev.nn; ++node)
  {
    Real grad_dir = 0.0;
    for (Index d = 0; d < ev.dim; ++d)
    {
      grad_dir += dir[d] * ev.grad(iq, node, d);
    }
    sum += absoluteValue(grad_dir);
  }

  return sum > 1.0e-14 ? 2.0 / sum : 0.0;
}

void stabilization(const QPState&     qp,
                   const KernelFluid& fluid,
                   Real               dt,
                   Real               h,
                   Index              dim,
                   Real               tau[3])
{
  Real speed2 = 0.0;
  for (Index d = 0; d < dim; ++d)
  {
    speed2 += qp.u[d] * qp.u[d];
  }

  const Real speed = sqrt(speed2);
  const Real nu    = fluid.mu / fluid.rho;
  const Real time  = (2.0 / dt) * (2.0 / dt);
  Real       flow  = 0.0;
  Real       diff  = 0.0;
  if (h > 0.0)
  {
    flow = (2.0 * speed / h) * (2.0 * speed / h);
    diff = (4.0 * nu / (h * h)) * (4.0 * nu / (h * h));
  }

  const Real value = 1.0 / sqrt(time + flow + diff);
  for (Index d = 0; d < 3; ++d)
  {
    tau[d] = value;
  }
}

} // namespace

Index vdof(Index node, Index comp, Index dim)
{
  return dim * node + comp;
}

Index pdof(Index node, Index nn, Index dim)
{
  return dim * nn + node;
}

Index numLocalDofs(Index nn, Index dim)
{
  return dim * nn + nn;
}

void zeroLocalSystem(Index nd, LocalMatrix Ke, LocalVector Fe)
{
  for (Index i = 0; i < nd; ++i)
  {
    Fe[i] = 0.0;
    for (Index j = 0; j < nd; ++j)
    {
      Ke(i, j) = 0.0;
    }
  }
}

void updateQpStates(const LocalElementValues& ev,
                    const KernelFluid&        fluid,
                    Real                      dt,
                    const Real*               state,
                    const Real*               adv_state,
                    QPState*                  qps)
{
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    QPState& qp = qps[iq];
    evalQp(ev, iq, state, adv_state, qp);
    stabilization(qp, fluid, dt, elemLength(ev, iq, qp.u), ev.dim, qp.tau);
  }
}

void assembleMassLHS(const LocalElementValues& ev,
                     const KernelFluid&        fluid,
                     Real                      dt,
                     LocalMatrix               Ke)
{
  const Real coeff = fluid.rho / dt;
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    const Real Jw = ev.wt(iq);
    for (Index i = 0; i < ev.nn; ++i)
    {
      const Real row = coeff * ev.shape(iq, i) * Jw;
      for (Index j = 0; j < ev.nn; ++j)
      {
        const Real value = row * ev.shape(iq, j);
        for (Index d = 0; d < ev.dim; ++d)
        {
          Ke(vdof(i, d, ev.dim), vdof(j, d, ev.dim)) += value;
        }
      }
    }
  }
}

void assembleMassRHS(const LocalElementValues& ev,
                     const QPState*            qps,
                     const KernelFluid&        fluid,
                     Real                      dt,
                     LocalVector               Fe)
{
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    const QPState& qp = qps[iq];
    const Real     Jw = ev.wt(iq);
    for (Index i = 0; i < ev.nn; ++i)
    {
      for (Index d = 0; d < ev.dim; ++d)
      {
        Fe[vdof(i, d, ev.dim)] +=
            fluid.rho / dt * ev.shape(iq, i) * qp.u[d] * Jw;
      }
    }
  }
}

void assembleAdvectionLHS(const LocalElementValues& ev,
                          const QPState*            qps,
                          const KernelFluid&        fluid,
                          LocalMatrix               Ke)
{
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    const QPState& qp = qps[iq];
    const Real     Jw = ev.wt(iq);
    for (Index i = 0; i < ev.nn; ++i)
    {
      const Real test = 0.5 * fluid.rho * ev.shape(iq, i) * Jw;
      for (Index j = 0; j < ev.nn; ++j)
      {
        const Real grad  = advectionDerivative(ev, qp, iq, j);
        const Real value = test * grad;
        for (Index d = 0; d < ev.dim; ++d)
        {
          Ke(vdof(i, d, ev.dim), vdof(j, d, ev.dim)) += value;
        }
      }
    }
  }
}

void assembleAdvectionRHS(const LocalElementValues& ev,
                          const QPState*            qps,
                          const KernelFluid&        fluid,
                          LocalVector               Fe)
{
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    const QPState& qp = qps[iq];
    const Real     Jw = ev.wt(iq);
    for (Index i = 0; i < ev.nn; ++i)
    {
      for (Index d = 0; d < ev.dim; ++d)
      {
        Fe[vdof(i, d, ev.dim)] -=
            0.5 * fluid.rho * ev.shape(iq, i) * qp.u_adv_grad_u[d] * Jw;
      }
    }
  }
}

void assembleDiffusionLHS(const LocalElementValues& ev,
                          const KernelFluid&        fluid,
                          LocalMatrix               Ke)
{
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    const Real Jw = ev.wt(iq);
    for (Index i = 0; i < ev.nn; ++i)
    {
      for (Index j = 0; j < ev.nn; ++j)
      {
        const Real value = 0.5 * fluid.mu * shapeGradientDot(ev, iq, i, j) * Jw;
        for (Index d = 0; d < ev.dim; ++d)
        {
          Ke(vdof(i, d, ev.dim), vdof(j, d, ev.dim)) += value;
        }
      }
    }
  }
}

void assembleDiffusionRHS(const LocalElementValues& ev,
                          const QPState*            qps,
                          const KernelFluid&        fluid,
                          LocalVector               Fe)
{
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    const QPState& qp = qps[iq];
    const Real     Jw = ev.wt(iq);
    for (Index i = 0; i < ev.nn; ++i)
    {
      for (Index c = 0; c < ev.dim; ++c)
      {
        Real dot = 0.0;
        for (Index d = 0; d < ev.dim; ++d)
        {
          dot += ev.grad(iq, i, d) * qp.grad_u[c][d];
        }
        Fe[vdof(i, c, ev.dim)] -= 0.5 * fluid.mu * dot * Jw;
      }
    }
  }
}

void assemblePreVelCouplingLHS(const LocalElementValues& ev,
                               LocalMatrix               Ke)
{
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    const Real Jw = ev.wt(iq);
    for (Index i = 0; i < ev.nn; ++i)
    {
      const Index ip = pdof(i, ev.nn, ev.dim);
      for (Index j = 0; j < ev.nn; ++j)
      {
        const Index jp = pdof(j, ev.nn, ev.dim);
        for (Index d = 0; d < ev.dim; ++d)
        {
          Ke(vdof(i, d, ev.dim), jp) -= ev.grad(iq, i, d) * ev.shape(iq, j) * Jw;
          Ke(ip, vdof(j, d, ev.dim)) += ev.shape(iq, i) * ev.grad(iq, j, d) * Jw;
        }
      }
    }
  }
}

void assembleStabilizationLHS(const LocalElementValues& ev,
                              const QPState*            qps,
                              const KernelFluid&        fluid,
                              Real                      dt,
                              LocalMatrix               Ke)
{
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    const QPState& qp = qps[iq];
    const Real     Jw = ev.wt(iq);
    for (Index i = 0; i < ev.nn; ++i)
    {
      const Index ip    = pdof(i, ev.nn, ev.dim);
      const Real  dvidx = advectionDerivative(ev, qp, iq, i);

      for (Index j = 0; j < ev.nn; ++j)
      {
        const Index jp    = pdof(j, ev.nn, ev.dim);
        const Real  dvjdx = advectionDerivative(ev, qp, iq, j);

        for (Index d = 0; d < ev.dim; ++d)
        {
          const Index iu = vdof(i, d, ev.dim);
          const Index ju = vdof(j, d, ev.dim);

          Ke(iu, ju) += qp.tau[0] * fluid.rho / dt * dvidx * ev.shape(iq, j) * Jw;
          Ke(iu, ju) += 0.5 * qp.tau[0] * fluid.rho * dvidx * dvjdx * Jw;
          Ke(iu, jp) += qp.tau[0] * dvidx * ev.grad(iq, j, d) * Jw;
          Ke(ip, ju) += qp.tau[1] / dt * ev.grad(iq, i, d) * ev.shape(iq, j) * Jw;
          Ke(ip, ju) += 0.5 * qp.tau[1] * ev.grad(iq, i, d) * dvjdx * Jw;
        }

        Ke(ip, jp) += qp.tau[1] / fluid.rho * shapeGradientDot(ev, iq, i, j)
                      * Jw;
      }
    }
  }
}

void assembleStabilizationRHS(const LocalElementValues& ev,
                              const QPState*            qps,
                              const KernelFluid&        fluid,
                              Real                      dt,
                              LocalVector               Fe)
{
  for (Index iq = 0; iq < ev.nq; ++iq)
  {
    const QPState& qp = qps[iq];
    const Real     Jw = ev.wt(iq);
    for (Index i = 0; i < ev.nn; ++i)
    {
      const Index ip             = pdof(i, ev.nn, ev.dim);
      const Real  dvidx          = advectionDerivative(ev, qp, iq, i);
      Real        div_u          = 0.0;
      Real        div_adv_grad_u = 0.0;

      for (Index d = 0; d < ev.dim; ++d)
      {
        const Index iu  = vdof(i, d, ev.dim);
        Fe[iu]         += qp.tau[0] * fluid.rho / dt * dvidx * qp.u[d] * Jw;
        Fe[iu]         -= 0.5 * qp.tau[0] * fluid.rho * dvidx * qp.u_adv_grad_u[d] * Jw;

        div_u          += ev.grad(iq, i, d) * qp.u[d];
        div_adv_grad_u += ev.grad(iq, i, d) * qp.u_adv_grad_u[d];
      }

      Fe[ip] += qp.tau[1] / dt * div_u * Jw;
      Fe[ip] -= 0.5 * qp.tau[1] * div_adv_grad_u * Jw;
    }
  }
}

void finishLocalResidual(Index       nd,
                         const Real* nxt,
                         LocalMatrix Ke,
                         LocalVector Fe,
                         Real*       out)
{
  for (Index i = 0; i < nd; ++i)
  {
    Real value = -Fe[i];
    for (Index j = 0; j < nd; ++j)
    {
      value += Ke(i, j) * nxt[j];
    }
    out[i] = value;
  }
}

} // namespace femx::navier
