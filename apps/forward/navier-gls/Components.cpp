#include "Components.hpp"

#include <femx/fem/ElementValues.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

Real advectionDerivative(const ElementValues& ev,
                         const QPState&       qp,
                         Index                iq,
                         Index                in)
{
  const auto  dNdx  = ev.dNdx(iq);
  const Index nd    = ev.dim();
  Real        value = 0.0;

  for (Index d = 0; d < nd; ++d)
  {
    value += dNdx(in, d) * qp.u_adv[d];
  }
  return value;
}

void assembleMassLHS(const ElementValues& ev,
                     const FluidParams&   fluid,
                     Real                 dt,
                     DenseMatrix&         Ke)
{
  const Index nshape = ev.numDofs();
  const Index nd     = ev.dim();
  const Index nq     = ev.numQuadraturePoints();
  const Real  coeff  = fluid.rho / dt;

  for (Index iq = 0; iq < nq; ++iq)
  {
    const auto N  = ev.N(iq);
    const Real Jw = ev.JxW(iq);

    for (Index i = 0; i < nshape; ++i)
    {
      const Real row = coeff * N[i] * Jw;
      for (Index j = 0; j < nshape; ++j)
      {
        const Real value = row * N[j];
        for (Index d = 0; d < nd; ++d)
        {
          Ke(nd * i + d, nd * j + d) += value;
        }
      }
    }
  }
}

void assembleMassRHS(const ElementValues&        ev,
                     const std::vector<QPState>& qps,
                     const FluidParams&          fluid,
                     Real                        dt,
                     Vector&                     Fe)
{
  const Index nshape = ev.numDofs();
  const Index nd     = ev.dim();
  const Index nq     = ev.numQuadraturePoints();

  for (Index iq = 0; iq < nq; ++iq)
  {
    const auto& qp = qps[static_cast<std::size_t>(iq)];
    const auto  N  = ev.N(iq);
    const auto  Jw = ev.JxW(iq);

    for (Index i = 0; i < nshape; ++i)
    {
      for (Index d = 0; d < nd; ++d)
      {
        Fe[nd * i + d] += fluid.rho / dt * N[i] * qp.u[d] * Jw;
      }
    }
  }
}

void assembleAdvectionLHS(const ElementValues&        ev,
                          const std::vector<QPState>& qps,
                          const FluidParams&          fluid,
                          DenseMatrix&                Ke)
{
  const Index nshape = ev.numDofs();
  const Index nd     = ev.dim();
  const Index nq     = ev.numQuadraturePoints();

  for (Index iq = 0; iq < nq; ++iq)
  {
    const auto& qp   = qps[static_cast<std::size_t>(iq)];
    const auto  N    = ev.N(iq);
    const auto  dNdx = ev.dNdx(iq);
    const Real  Jw   = ev.JxW(iq);

    for (Index i = 0; i < nshape; ++i)
    {
      const Real test = 0.5 * fluid.rho * N[i] * Jw;
      for (Index j = 0; j < nshape; ++j)
      {
        Real grad = 0.0;
        for (Index d = 0; d < nd; ++d)
        {
          grad += qp.u_adv[d] * dNdx(j, d);
        }

        const Real value = test * grad;
        for (Index d = 0; d < nd; ++d)
        {
          Ke(nd * i + d, nd * j + d) += value;
        }
      }
    }
  }
}

void assembleAdvectionRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qps,
                          const FluidParams&          fluid,
                          Vector&                     Fe)
{
  const Index nshape = ev.numDofs();
  const Index nd     = ev.dim();
  const Index nq     = ev.numQuadraturePoints();

  for (Index iq = 0; iq < nq; ++iq)
  {
    const auto& qp = qps[static_cast<std::size_t>(iq)];
    const auto  N  = ev.N(iq);
    const auto  Jw = ev.JxW(iq);
    for (Index i = 0; i < nshape; ++i)
    {
      for (Index d = 0; d < nd; ++d)
      {
        Fe[nd * i + d] -=
            0.5 * fluid.rho * N[i] * qp.u_adv_grad_u[d] * Jw;
      }
    }
  }
}

void assembleDiffusionLHS(const ElementValues& ev,
                          const FluidParams&   fluid,
                          DenseMatrix&         Ke)
{
  const Index nshape = ev.numDofs();
  const Index nd     = ev.dim();
  const Index nq     = ev.numQuadraturePoints();

  for (Index iq = 0; iq < nq; ++iq)
  {
    const auto dNdx = ev.dNdx(iq);
    const Real Jw   = ev.JxW(iq);

    for (Index i = 0; i < nshape; ++i)
    {
      for (Index j = 0; j < nshape; ++j)
      {
        Real dot = 0.0;
        for (Index d = 0; d < nd; ++d)
        {
          dot += dNdx(i, d) * dNdx(j, d);
        }

        const Real value = 0.5 * fluid.mu * dot * Jw;
        for (Index d = 0; d < nd; ++d)
        {
          Ke(nd * i + d, nd * j + d) += value;
        }
      }
    }
  }
}

void assembleDiffusionRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qps,
                          const FluidParams&          fluid,
                          Vector&                     Fe)
{
  const Index nshape = ev.numDofs();
  const Index nd     = ev.dim();
  const Index nq     = ev.numQuadraturePoints();

  for (Index iq = 0; iq < nq; ++iq)
  {
    const auto& qp   = qps[static_cast<std::size_t>(iq)];
    const auto  dNdx = ev.dNdx(iq);
    const auto  Jw   = ev.JxW(iq);

    for (Index i = 0; i < nshape; ++i)
    {
      for (Index c = 0; c < nd; ++c)
      {
        Real dot = 0.0;
        for (Index d = 0; d < nd; ++d)
        {
          dot += dNdx(i, d) * qp.grad_u[c][d];
        }
        Fe[ev.dim() * i + c] -= 0.5 * fluid.mu * dot * Jw;
      }
    }
  }
}

void assemblePreVelCouplingLHS(const ElementValues& ev,
                               DenseMatrix&         Ke)
{
  const Index nshape = ev.numDofs();
  const Index nd     = ev.dim();

  for (Index iq = 0; iq < ev.numQuadraturePoints(); ++iq)
  {
    const auto N    = ev.N(iq);
    const auto dNdx = ev.dNdx(iq);
    const auto Jw   = ev.JxW(iq);

    for (Index i = 0; i < nshape; ++i)
    {
      const Index ip = nd * nshape + i;

      for (Index j = 0; j < nshape; ++j)
      {
        const Index jp = nd * nshape + j;
        for (Index d = 0; d < nd; ++d)
        {
          Ke(nd * i + d, jp) -= dNdx(i, d) * N[j] * Jw;
          Ke(ip, nd * j + d) += N[i] * dNdx(j, d) * Jw;
        }
      }
    }
  }
}

void assembleStabilizationLHS(const ElementValues&        ev,
                              const std::vector<QPState>& qps,
                              const FluidParams&          fluid,
                              Real                        dt,
                              DenseMatrix&                Ke)
{
  const Index nshape = ev.numDofs();
  const Index nd     = ev.dim();

  for (Index iq = 0; iq < ev.numQuadraturePoints(); ++iq)
  {
    const auto& qp   = qps[static_cast<std::size_t>(iq)];
    const auto  N    = ev.N(iq);
    const auto  dNdx = ev.dNdx(iq);
    const auto  Jw   = ev.JxW(iq);

    for (Index i = 0; i < nshape; ++i)
    {
      const Index ip    = nd * nshape + i;
      const Real  dvidx = advectionDerivative(ev, qp, iq, i);

      for (Index j = 0; j < nshape; ++j)
      {
        const Index jp    = nd * nshape + j;
        const Real  dvjdx = advectionDerivative(ev, qp, iq, j);

        for (Index d = 0; d < nd; ++d)
        {
          const Index iu      = nd * i + d;
          const Index ju      = nd * j + d;
          Ke(iu, ju)         += qp.tau[0] * fluid.rho / dt * dvidx * N[j] * Jw;
          Ke(iu, ju)         += 0.5 * qp.tau[0] * fluid.rho * dvidx * dvjdx * Jw;
          Ke(iu, jp)         += qp.tau[0] * dvidx * dNdx(j, d) * Jw;
          Ke(ip, nd * j + d) += qp.tau[1] / dt * dNdx(i, d) * N[j] * Jw;
          Ke(ip, nd * j + d) += 0.5 * qp.tau[1] * dNdx(i, d) * dvjdx * Jw;
        }

        Real dot = 0.0;
        for (Index d = 0; d < nd; ++d)
        {
          dot += dNdx(i, d) * dNdx(j, d);
        }
        Ke(ip, jp) += qp.tau[1] / fluid.rho * dot * Jw;
      }
    }
  }
}

void assembleStabilizationRHS(const ElementValues&        ev,
                              const std::vector<QPState>& qps,
                              const FluidParams&          fluid,
                              Real                        dt,
                              Vector&                     Fe)
{
  const Index nshape = ev.numDofs();
  const Index nd     = ev.dim();

  for (Index iq = 0; iq < ev.numQuadraturePoints(); ++iq)
  {
    const auto& qp   = qps[static_cast<std::size_t>(iq)];
    const auto  dNdx = ev.dNdx(iq);
    const auto  Jw   = ev.JxW(iq);

    for (Index i = 0; i < nshape; ++i)
    {
      const Index ip = nd * nshape + i;

      const Real dvidx = advectionDerivative(ev, qp, iq, i);

      Real div_u          = 0.0;
      Real div_adv_grad_u = 0.0;
      for (Index d = 0; d < nd; ++d)
      {
        const Index iu  = nd * i + d;
        Fe[iu]         += qp.tau[0] * fluid.rho / dt * dvidx * qp.u[d] * Jw;
        Fe[iu]         -= 0.5 * qp.tau[0] * fluid.rho * dvidx * qp.u_adv_grad_u[d] * Jw;
        div_u          += dNdx(i, d) * qp.u[d];
        div_adv_grad_u += dNdx(i, d) * qp.u_adv_grad_u[d];
      }

      Fe[ip] += qp.tau[1] / dt * div_u * Jw;
      Fe[ip] -= 0.5 * qp.tau[1] * div_adv_grad_u * Jw;
    }
  }
}

} // namespace femx
