#include "Components.hpp"

#include <femx/fem/ElementValues.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

real_type advectionDerivative(const ElementValues& ev,
                              const QPState&       qp,
                              index_type           q,
                              index_type           node)
{
  const auto       dNdx  = ev.dNdx(q);
  const index_type nd    = ev.dim();
  real_type        value = 0.0;

  for (index_type d = 0; d < nd; ++d)
  {
    value += dNdx(node, d) * qp.u_adv[d];
  }
  return value;
}

void assembleMassLHS(const ElementValues& ev,
                     const FluidParams&   fluid,
                     real_type            dt,
                     DenseMatrix&         Ke)
{
  const index_type nshape = ev.numDofs();
  const index_type nd     = ev.dim();
  const index_type nq     = ev.numQuadraturePoints();
  const real_type  coeff  = fluid.rho / dt;

  for (index_type q = 0; q < nq; ++q)
  {
    const auto      N  = ev.N(q);
    const real_type Jw = ev.JxW(q);

    for (index_type i = 0; i < nshape; ++i)
    {
      const real_type row = coeff * N[i] * Jw;
      for (index_type j = 0; j < nshape; ++j)
      {
        const real_type value = row * N[j];
        for (index_type d = 0; d < nd; ++d)
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
                     real_type                   dt,
                     Vector&                     Fe)
{
  const index_type nshape = ev.numDofs();
  const index_type nd     = ev.dim();
  const index_type nq     = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto& qp = qps[static_cast<std::size_t>(q)];
    const auto  N  = ev.N(q);
    const auto  Jw = ev.JxW(q);

    for (index_type i = 0; i < nshape; ++i)
    {
      for (index_type d = 0; d < nd; ++d)
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
  const index_type nshape = ev.numDofs();
  const index_type nd     = ev.dim();
  const index_type nq     = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto&     qp   = qps[static_cast<std::size_t>(q)];
    const auto      N    = ev.N(q);
    const auto      dNdx = ev.dNdx(q);
    const real_type Jw   = ev.JxW(q);

    for (index_type i = 0; i < nshape; ++i)
    {
      const real_type test = 0.5 * fluid.rho * N[i] * Jw;
      for (index_type j = 0; j < nshape; ++j)
      {
        real_type grad = 0.0;
        for (index_type d = 0; d < nd; ++d)
        {
          grad += qp.u_adv[d] * dNdx(j, d);
        }

        const real_type value = test * grad;
        for (index_type d = 0; d < nd; ++d)
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
  const index_type nshape = ev.numDofs();
  const index_type nd     = ev.dim();
  const index_type nq     = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto& qp = qps[static_cast<std::size_t>(q)];
    const auto  N  = ev.N(q);
    const auto  Jw = ev.JxW(q);
    for (index_type i = 0; i < nshape; ++i)
    {
      for (index_type d = 0; d < nd; ++d)
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
  const index_type nshape = ev.numDofs();
  const index_type nd     = ev.dim();
  const index_type nq     = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto      dNdx = ev.dNdx(q);
    const real_type Jw   = ev.JxW(q);

    for (index_type i = 0; i < nshape; ++i)
    {
      for (index_type j = 0; j < nshape; ++j)
      {
        real_type dot = 0.0;
        for (index_type d = 0; d < nd; ++d)
        {
          dot += dNdx(i, d) * dNdx(j, d);
        }

        const real_type value = 0.5 * fluid.mu * dot * Jw;
        for (index_type d = 0; d < nd; ++d)
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
  const index_type nshape = ev.numDofs();
  const index_type nd     = ev.dim();
  const index_type nq     = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto& qp   = qps[static_cast<std::size_t>(q)];
    const auto  dNdx = ev.dNdx(q);
    const auto  Jw   = ev.JxW(q);

    for (index_type i = 0; i < nshape; ++i)
    {
      for (index_type c = 0; c < nd; ++c)
      {
        real_type dot = 0.0;
        for (index_type d = 0; d < nd; ++d)
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
  const index_type nshape = ev.numDofs();
  const index_type nd     = ev.dim();

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto N    = ev.N(q);
    const auto dNdx = ev.dNdx(q);
    const auto Jw   = ev.JxW(q);

    for (index_type i = 0; i < nshape; ++i)
    {
      const index_type ip = nd * nshape + i;

      for (index_type j = 0; j < nshape; ++j)
      {
        const index_type jp = nd * nshape + j;
        for (index_type d = 0; d < nd; ++d)
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
                              real_type                   dt,
                              DenseMatrix&                Ke)
{
  const index_type nshape = ev.numDofs();
  const index_type nd     = ev.dim();

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& qp   = qps[static_cast<std::size_t>(q)];
    const auto  N    = ev.N(q);
    const auto  dNdx = ev.dNdx(q);
    const auto  Jw   = ev.JxW(q);

    for (index_type i = 0; i < nshape; ++i)
    {
      const index_type ip    = nd * nshape + i;
      const real_type  dvidx = advectionDerivative(ev, qp, q, i);

      for (index_type j = 0; j < nshape; ++j)
      {
        const index_type jp    = nd * nshape + j;
        const real_type  dvjdx = advectionDerivative(ev, qp, q, j);

        for (index_type d = 0; d < nd; ++d)
        {
          const index_type iu  = nd * i + d;
          const index_type ju  = nd * j + d;
          Ke(iu, ju)          += qp.tau[0] * fluid.rho / dt * dvidx * N[j] * Jw;
          Ke(iu, ju)          += 0.5 * qp.tau[0] * fluid.rho * dvidx * dvjdx * Jw;
          Ke(iu, jp)          += qp.tau[0] * dvidx * dNdx(j, d) * Jw;
          Ke(ip, nd * j + d)  += qp.tau[1] / dt * dNdx(i, d) * N[j] * Jw;
          Ke(ip, nd * j + d)  += 0.5 * qp.tau[1] * dNdx(i, d) * dvjdx * Jw;
        }

        real_type dot = 0.0;
        for (index_type d = 0; d < nd; ++d)
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
                              real_type                   dt,
                              Vector&                     Fe)
{
  const index_type nshape = ev.numDofs();
  const index_type nd     = ev.dim();

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& qp   = qps[static_cast<std::size_t>(q)];
    const auto  dNdx = ev.dNdx(q);
    const auto  Jw   = ev.JxW(q);

    for (index_type i = 0; i < nshape; ++i)
    {
      const index_type ip = nd * nshape + i;

      const real_type dvidx = advectionDerivative(ev, qp, q, i);

      real_type div_u          = 0.0;
      real_type div_adv_grad_u = 0.0;
      for (index_type d = 0; d < nd; ++d)
      {
        const index_type iu  = nd * i + d;
        Fe[iu]              += qp.tau[0] * fluid.rho / dt * dvidx * qp.u[d] * Jw;
        Fe[iu]              -= 0.5 * qp.tau[0] * fluid.rho * dvidx * qp.u_adv_grad_u[d] * Jw;
        div_u               += dNdx(i, d) * qp.u[d];
        div_adv_grad_u      += dNdx(i, d) * qp.u_adv_grad_u[d];
      }

      Fe[ip] += qp.tau[1] / dt * div_u * Jw;
      Fe[ip] -= 0.5 * qp.tau[1] * div_adv_grad_u * Jw;
    }
  }
}

} // namespace femx
