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
  const auto dNdx  = ev.dNdx(q);
  real_type  value = 0.0;
  for (index_type d = 0; d < ev.dim(); ++d)
  {
    value += dNdx(node, d) * qp.u_adv[d];
  }
  return value;
}

void assembleTransientLHS(const ElementValues& ev,
                          DenseMatrix&         Ke)
{
  const index_type num_shapes = ev.numDofs();
  const index_type nd         = ev.dim();
  const index_type nq         = ev.numQuadraturePoints();
  const real_type  coeff      = rho / dt;

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
        for (index_type c = 0; c < nd; ++c)
        {
          Ke(nd * i + c, nd * j + c) += value;
        }
      }
    }
  }
}

void assembleTransientRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          Vector&                     Fe)
{
  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& qp = qp_states[static_cast<std::size_t>(q)];
    const auto  N  = ev.N(q);
    const auto  wJ = ev.JxW(q);
    for (index_type i = 0; i < ev.numDofs(); ++i)
    {
      for (index_type c = 0; c < ev.dim(); ++c)
      {
        Fe[ev.dim() * i + c] += rho / dt * N[i] * qp.u[c] * wJ;
      }
    }
  }
}

void assembleAdvectionLHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
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
      const real_type row = 0.5 * rho * N[i] * wJ;
      for (index_type j = 0; j < num_shapes; ++j)
      {
        real_type adv_grad = 0.0;
        for (index_type d = 0; d < nd; ++d)
        {
          adv_grad += qp.u_adv[d] * dNdx(j, d);
        }

        const real_type value = row * adv_grad;
        for (index_type c = 0; c < nd; ++c)
        {
          Ke(nd * i + c, nd * j + c) += value;
        }
      }
    }
  }
}

void assembleAdvectionRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          Vector&                     Fe)
{
  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& qp = qp_states[static_cast<std::size_t>(q)];
    const auto  N  = ev.N(q);
    const auto  wJ = ev.JxW(q);
    for (index_type i = 0; i < ev.numDofs(); ++i)
    {
      for (index_type c = 0; c < ev.dim(); ++c)
      {
        Fe[ev.dim() * i + c] -=
            0.5 * rho * N[i] * qp.u_adv_grad_u[c] * wJ;
      }
    }
  }
}

void assembleViscousLHS(const ElementValues& ev,
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
        real_type grad_dot = 0.0;
        for (index_type d = 0; d < nd; ++d)
        {
          grad_dot += dNdx(i, d) * dNdx(j, d);
        }

        const real_type value = 0.5 * mu * grad_dot * wJ;
        for (index_type c = 0; c < nd; ++c)
        {
          Ke(nd * i + c, nd * j + c) += value;
        }
      }
    }
  }
}

void assembleViscousRHS(const ElementValues&        ev,
                        const std::vector<QPState>& qp_states,
                        Vector&                     Fe)
{
  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& qp   = qp_states[static_cast<std::size_t>(q)];
    const auto  dNdx = ev.dNdx(q);
    const auto  wJ   = ev.JxW(q);

    for (index_type i = 0; i < ev.numDofs(); ++i)
    {
      for (index_type c = 0; c < ev.dim(); ++c)
      {
        real_type grad_dot = 0.0;
        for (index_type d = 0; d < ev.dim(); ++d)
        {
          grad_dot += dNdx(i, d) * qp.grad_u[c][d];
        }
        Fe[ev.dim() * i + c] -= 0.5 * mu * grad_dot * wJ;
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
        for (index_type c = 0; c < nd; ++c)
        {
          Ke(nd * i + c, jp) -= dNdx(i, c) * N[j] * wJ;
          Ke(ip, nd * j + c) += N[i] * dNdx(j, c) * wJ;
        }
      }
    }
  }
}

void assembleStabilizationLHS(const ElementValues&        ev,
                              const std::vector<QPState>& qp_states,
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

        for (index_type c = 0; c < nd; ++c)
        {
          const index_type iu  = nd * i + c;
          const index_type ju  = nd * j + c;
          Ke(iu, ju)          += qp.tau[0] * rho / dt * dvidx * N[j] * wJ;
          Ke(iu, ju)          += 0.5 * qp.tau[0] * rho * dvidx * dvjdx * wJ;
          Ke(iu, jp)          += qp.tau[0] * dvidx * dNdx(j, c) * wJ;
          Ke(ip, nd * j + c)  += qp.tau[1] / dt * dNdx(i, c) * N[j] * wJ;
          Ke(ip, nd * j + c)  += 0.5 * qp.tau[1] * dNdx(i, c) * dvjdx * wJ;
        }

        real_type grad_dot = 0.0;
        for (index_type c = 0; c < nd; ++c)
        {
          grad_dot += dNdx(i, c) * dNdx(j, c);
        }
        Ke(ip, jp) += qp.tau[1] / rho * grad_dot * wJ;
      }
    }
  }
}

void assembleStabilizationRHS(const ElementValues&        ev,
                              const std::vector<QPState>& qp_states,
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
      for (index_type c = 0; c < nd; ++c)
      {
        const index_type iu  = nd * i + c;
        Fe[iu]              += qp.tau[0] * rho / dt * dvidx * qp.u[c] * wJ;
        Fe[iu]              -= 0.5 * qp.tau[0] * rho * dvidx * qp.u_adv_grad_u[c] * wJ;
        div_u               += dNdx(i, c) * qp.u[c];
        div_adv_grad_u      += dNdx(i, c) * qp.u_adv_grad_u[c];
      }

      Fe[ip] += qp.tau[1] / dt * div_u * wJ;
      Fe[ip] -= 0.5 * qp.tau[1] * div_adv_grad_u * wJ;
    }
  }
}

} // namespace refem
