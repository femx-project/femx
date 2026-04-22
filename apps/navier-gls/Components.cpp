#include "Components.hpp"

#include <refem/fe/ElementValues.hpp>
#include <refem/linalg/DenseMatrix.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
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
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();
  const index_type nq         = ev.numQuadraturePoints();
  const real_type  coeff      = fluid.rho / dt;

  for (index_type q = 0; q < nq; ++q)
  {
    const auto      N  = ev.N(q);
    const real_type wJ = ev.JxW(q);

    for (index_type i = 0; i < num_shapes; ++i)
    {
      const real_type row = coeff * N[i] * wJ;
      for (index_type j = 0; j < num_shapes; ++j)
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
                     const std::vector<QPState>& qp_states,
                     const FluidParams&          fluid,
                     real_type                   dt,
                     Vector&                     Fe)
{
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();
  const index_type nq         = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto& qp = qp_states[static_cast<std::size_t>(q)];
    const auto  N  = ev.N(q);
    const auto  wJ = ev.JxW(q);

    for (index_type i = 0; i < num_shapes; ++i)
    {
      for (index_type d = 0; d < nd; ++d)
      {
        Fe[nd * i + d] += fluid.rho / dt * N[i] * qp.u[d] * wJ;
      }
    }
  }
}

void assembleAdvectionLHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          const FluidParams&          fluid,
                          DenseMatrix&                Ke)
{
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();
  const index_type nq         = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto&     qp   = qp_states[static_cast<std::size_t>(q)];
    const auto      N    = ev.N(q);
    const auto      dNdx = ev.dNdx(q);
    const real_type wJ   = ev.JxW(q);

    for (index_type i = 0; i < num_shapes; ++i)
    {
      const real_type row = 0.5 * fluid.rho * N[i] * wJ;
      for (index_type j = 0; j < num_shapes; ++j)
      {
        real_type adv_grad = 0.0;
        for (index_type d = 0; d < nd; ++d)
        {
          adv_grad += qp.u_adv[d] * dNdx(j, d);
        }

        const real_type value = row * adv_grad;
        for (index_type d = 0; d < nd; ++d)
        {
          Ke(nd * i + d, nd * j + d) += value;
        }
      }
    }
  }
}

void assembleAdvectionRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          const FluidParams&          fluid,
                          Vector&                     Fe)
{
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();
  const index_type nq         = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto& qp = qp_states[static_cast<std::size_t>(q)];
    const auto  N  = ev.N(q);
    const auto  wJ = ev.JxW(q);
    for (index_type i = 0; i < num_shapes; ++i)
    {
      for (index_type d = 0; d < nd; ++d)
      {
        Fe[nd * i + d] -=
            0.5 * fluid.rho * N[i] * qp.u_adv_grad_u[d] * wJ;
      }
    }
  }
}

void assembleDiffusionLHS(const ElementValues& ev,
                          const FluidParams&   fluid,
                          DenseMatrix&         Ke)
{
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();
  const index_type nq         = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto      dNdx = ev.dNdx(q);
    const real_type wJ   = ev.JxW(q);

    for (index_type i = 0; i < num_shapes; ++i)
    {
      for (index_type j = 0; j < num_shapes; ++j)
      {
        real_type dot = 0.0;
        for (index_type d = 0; d < nd; ++d)
        {
          dot += dNdx(i, d) * dNdx(j, d);
        }

        const real_type value = 0.5 * fluid.mu * dot * wJ;
        for (index_type d = 0; d < nd; ++d)
        {
          Ke(nd * i + d, nd * j + d) += value;
        }
      }
    }
  }
}

void assembleDiffusionRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          const FluidParams&          fluid,
                          Vector&                     Fe)
{
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();
  const index_type nq         = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto& qp   = qp_states[static_cast<std::size_t>(q)];
    const auto  dNdx = ev.dNdx(q);
    const auto  wJ   = ev.JxW(q);

    for (index_type i = 0; i < num_shapes; ++i)
    {
      for (index_type c = 0; c < nd; ++c)
      {
        real_type dot = 0.0;
        for (index_type d = 0; d < nd; ++d)
        {
          dot += dNdx(i, d) * qp.grad_u[c][d];
        }
        Fe[ev.dim() * i + c] -= 0.5 * fluid.mu * dot * wJ;
      }
    }
  }
}

void assemblePressureVelocityCouplingLHS(const ElementValues& ev,
                                         DenseMatrix&         Ke)
{
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto N    = ev.N(q);
    const auto dNdx = ev.dNdx(q);
    const auto wJ   = ev.JxW(q);

    for (index_type i = 0; i < num_shapes; ++i)
    {
      const index_type ip = nd * num_shapes + i;

      for (index_type j = 0; j < num_shapes; ++j)
      {
        const index_type jp = nd * num_shapes + j;
        for (index_type d = 0; d < nd; ++d)
        {
          Ke(nd * i + d, jp) -= dNdx(i, d) * N[j] * wJ;
          Ke(ip, nd * j + d) += N[i] * dNdx(j, d) * wJ;
        }
      }
    }
  }
}

void assembleStabilizationLHS(const ElementValues&        ev,
                              const std::vector<QPState>& qp_states,
                              const FluidParams&          fluid,
                              real_type                   dt,
                              DenseMatrix&                Ke)
{
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& qp   = qp_states[static_cast<std::size_t>(q)];
    const auto  N    = ev.N(q);
    const auto  dNdx = ev.dNdx(q);
    const auto  wJ   = ev.JxW(q);

    for (index_type i = 0; i < num_shapes; ++i)
    {
      const index_type ip    = nd * num_shapes + i;
      const real_type  dvidx = advectionDerivative(ev, qp, q, i);

      for (index_type j = 0; j < num_shapes; ++j)
      {
        const index_type jp    = nd * num_shapes + j;
        const real_type  dvjdx = advectionDerivative(ev, qp, q, j);

        for (index_type d = 0; d < nd; ++d)
        {
          const index_type iu  = nd * i + d;
          const index_type ju  = nd * j + d;
          Ke(iu, ju)          += qp.tau[0] * fluid.rho / dt * dvidx * N[j] * wJ;
          Ke(iu, ju)          += 0.5 * qp.tau[0] * fluid.rho * dvidx * dvjdx * wJ;
          Ke(iu, jp)          += qp.tau[0] * dvidx * dNdx(j, d) * wJ;
          Ke(ip, nd * j + d)  += qp.tau[1] / dt * dNdx(i, d) * N[j] * wJ;
          Ke(ip, nd * j + d)  += 0.5 * qp.tau[1] * dNdx(i, d) * dvjdx * wJ;
        }

        real_type dot = 0.0;
        for (index_type d = 0; d < nd; ++d)
        {
          dot += dNdx(i, d) * dNdx(j, d);
        }
        Ke(ip, jp) += qp.tau[1] / fluid.rho * dot * wJ;
      }
    }
  }
}

void assembleStabilizationRHS(const ElementValues&        ev,
                              const std::vector<QPState>& qp_states,
                              const FluidParams&          fluid,
                              real_type                   dt,
                              Vector&                     Fe)
{
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& qp   = qp_states[static_cast<std::size_t>(q)];
    const auto  dNdx = ev.dNdx(q);
    const auto  wJ   = ev.JxW(q);

    for (index_type i = 0; i < num_shapes; ++i)
    {
      const index_type ip = nd * num_shapes + i;

      const real_type dvidx = advectionDerivative(ev, qp, q, i);

      real_type div_u          = 0.0;
      real_type div_adv_grad_u = 0.0;
      for (index_type d = 0; d < nd; ++d)
      {
        const index_type iu  = nd * i + d;
        Fe[iu]              += qp.tau[0] * fluid.rho / dt * dvidx * qp.u[d] * wJ;
        Fe[iu]              -= 0.5 * qp.tau[0] * fluid.rho * dvidx * qp.u_adv_grad_u[d] * wJ;
        div_u               += dNdx(i, d) * qp.u[d];
        div_adv_grad_u      += dNdx(i, d) * qp.u_adv_grad_u[d];
      }

      Fe[ip] += qp.tau[1] / dt * div_u * wJ;
      Fe[ip] -= 0.5 * qp.tau[1] * div_adv_grad_u * wJ;
    }
  }
}

} // namespace refem
