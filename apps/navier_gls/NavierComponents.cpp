#include "NavierComponents.hpp"

#include <refem/fe/ElementValues.hpp>
#include <refem/forms/integrators/AdvectionIntegrator.hpp>
#include <refem/forms/integrators/DiffusionIntegrator.hpp>
#include <refem/forms/integrators/MassIntegrator.hpp>
#include <refem/linalg/DenseMatrix.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{

real_type advectionDerivative(const ElementValues& ev,
                              const QPState&       point,
                              index_type           q,
                              index_type           node)
{
  const auto dNdx = ev.dNdx(q);
  return dNdx(node, 0) * point.u_adv[0] + dNdx(node, 1) * point.u_adv[1];
}

void TransientLHS::assemble(const ElementValues& ev,
                            DenseMatrix&         Ke) const
{
  const index_type num_shapes = ev.numDofs();

  DenseMatrix Me(num_shapes, num_shapes);
  Me.setZero();
  MassIntegrator(rho / dt).assemble(ev, Me);

  for (index_type a = 0; a < num_shapes; ++a)
  {
    const index_type iu = dim * a;
    const index_type iv = iu + 1;

    for (index_type b = 0; b < num_shapes; ++b)
    {
      const index_type ju = dim * b;
      const index_type jv = ju + 1;

      Ke(iu, ju) += Me(a, b);
      Ke(iv, jv) += Me(a, b);
    }
  }
}

TransientRHS::TransientRHS(const std::vector<QPState>& state)
  : state_(state)
{
}

void TransientRHS::assemble(const ElementValues& ev,
                            Vector&              Fe) const
{
  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& point = state_[static_cast<std::size_t>(q)];
    const auto  N     = ev.N(q);
    const auto  wJ    = ev.JxW(q);
    for (index_type a = 0; a < ev.numDofs(); ++a)
    {
      const index_type iu = dim * a;
      const index_type iv = iu + 1;

      Fe[iu] += rho / dt * N[a] * point.u[0] * wJ;
      Fe[iv] += rho / dt * N[a] * point.u[1] * wJ;
    }
  }
}

AdvectionLHS::AdvectionLHS(const std::vector<QPState>& state)
  : state_(state)
{
}

void AdvectionLHS::assemble(const ElementValues& ev,
                            DenseMatrix&         Ke) const
{
  const index_type num_shapes = ev.numDofs();

  DenseMatrix Ce(num_shapes, num_shapes);
  Ce.setZero();

  std::vector<real_type> advection_velocity(
      static_cast<std::size_t>(ev.numQuadraturePoints() * dim), 0.0);
  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& point = state_[static_cast<std::size_t>(q)];
    for (index_type d = 0; d < dim; ++d)
    {
      advection_velocity[static_cast<std::size_t>(q * dim + d)] =
          point.u_adv[d];
    }
  }

  AdvectionIntegrator(advection_velocity, 0.5 * rho)
      .assemble(ev, Ce);

  for (index_type a = 0; a < num_shapes; ++a)
  {
    const index_type iu = dim * a;
    const index_type iv = iu + 1;

    for (index_type b = 0; b < num_shapes; ++b)
    {
      const index_type ju = dim * b;
      const index_type jv = ju + 1;

      Ke(iu, ju) += Ce(a, b);
      Ke(iv, jv) += Ce(a, b);
    }
  }
}

AdvectionRHS::AdvectionRHS(const std::vector<QPState>& state)
  : state_(state)
{
}

void AdvectionRHS::assemble(const ElementValues& ev,
                            Vector&              Fe) const
{
  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& point = state_[static_cast<std::size_t>(q)];
    const auto  N     = ev.N(q);
    const auto  wJ    = ev.JxW(q);
    for (index_type a = 0; a < ev.numDofs(); ++a)
    {
      const index_type iu = dim * a;
      const index_type iv = iu + 1;

      Fe[iu] -= 0.5 * rho * N[a] * point.grad_u_adv[0] * wJ;
      Fe[iv] -= 0.5 * rho * N[a] * point.grad_u_adv[1] * wJ;
    }
  }
}

void ViscousLHS::assemble(const ElementValues& ev,
                          DenseMatrix&         Ke) const
{
  const index_type num_shapes = ev.numDofs();

  DenseMatrix De(num_shapes, num_shapes);
  De.setZero();
  DiffusionIntegrator(0.5 * mu).assemble(ev, De);

  for (index_type a = 0; a < num_shapes; ++a)
  {
    const index_type iu = dim * a;
    const index_type iv = iu + 1;

    for (index_type b = 0; b < num_shapes; ++b)
    {
      const index_type ju = dim * b;
      const index_type jv = ju + 1;

      Ke(iu, ju) += De(a, b);
      Ke(iv, jv) += De(a, b);
    }
  }
}

ViscousRHS::ViscousRHS(const std::vector<QPState>& state)
  : state_(state)
{
}

void ViscousRHS::assemble(const ElementValues& ev,
                          Vector&              Fe) const
{
  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& point = state_[static_cast<std::size_t>(q)];
    const auto  dNdx  = ev.dNdx(q);
    const auto  wJ    = ev.JxW(q);

    for (index_type a = 0; a < ev.numDofs(); ++a)
    {
      const index_type iu = dim * a;
      const index_type iv = iu + 1;

      Fe[iu] -=
          0.5 * mu * (dNdx(a, 0) * point.grad[0][0] + dNdx(a, 1) * point.grad[0][1]) * wJ;
      Fe[iv] -=
          0.5 * mu * (dNdx(a, 0) * point.grad[1][0] + dNdx(a, 1) * point.grad[1][1]) * wJ;
    }
  }
}

void PressureVelocityCouplingLHS::assemble(const ElementValues& ev,
                                           DenseMatrix&         Ke) const
{
  const index_type num_shapes = ev.numDofs();

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto N    = ev.N(q);
    const auto dNdx = ev.dNdx(q);
    const auto wJ   = ev.JxW(q);

    for (index_type a = 0; a < num_shapes; ++a)
    {
      const index_type iu = dim * a;
      const index_type iv = iu + 1;
      const index_type ip = dim * num_shapes + a;

      for (index_type b = 0; b < num_shapes; ++b)
      {
        const index_type ju = dim * b;
        const index_type jv = ju + 1;
        const index_type jp = dim * num_shapes + b;

        Ke(iu, jp) -= dNdx(a, 0) * N[b] * wJ;
        Ke(iv, jp) -= dNdx(a, 1) * N[b] * wJ;

        Ke(ip, ju) += N[a] * dNdx(b, 0) * wJ;
        Ke(ip, jv) += N[a] * dNdx(b, 1) * wJ;
      }
    }
  }
}

StabilizationLHS::StabilizationLHS(const std::vector<QPState>& state)
  : state_(state)
{
}

void StabilizationLHS::assemble(const ElementValues& ev,
                                DenseMatrix&         Ke) const
{
  const index_type num_shapes = ev.numDofs();

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& point = state_[static_cast<std::size_t>(q)];
    const auto  N     = ev.N(q);
    const auto  dNdx  = ev.dNdx(q);
    const auto  wJ    = ev.JxW(q);

    for (index_type a = 0; a < num_shapes; ++a)
    {
      const index_type iu = dim * a;
      const index_type iv = iu + 1;
      const index_type ip = dim * num_shapes + a;

      const real_type advection_a = advectionDerivative(ev, point, q, a);

      for (index_type b = 0; b < num_shapes; ++b)
      {
        const index_type ju = dim * b;
        const index_type jv = ju + 1;
        const index_type jp = dim * num_shapes + b;

        const real_type advection_b = advectionDerivative(ev, point, q, b);

        Ke(iu, ju) +=
            point.tau[0] * rho / dt * advection_a * N[b] * wJ;
        Ke(iu, ju) +=
            0.5 * point.tau[0] * rho * advection_a * advection_b * wJ;
        Ke(iv, jv) +=
            point.tau[0] * rho / dt * advection_a * N[b] * wJ;
        Ke(iv, jv) +=
            0.5 * point.tau[0] * rho * advection_a * advection_b * wJ;

        Ke(iu, jp) += point.tau[0] * advection_a * dNdx(b, 0) * wJ;
        Ke(iv, jp) += point.tau[0] * advection_a * dNdx(b, 1) * wJ;

        Ke(ip, ju) +=
            point.tau[1] / dt * dNdx(a, 0) * N[b] * wJ;
        Ke(ip, ju) +=
            0.5 * point.tau[1] * dNdx(a, 0) * advection_b * wJ;
        Ke(ip, jv) +=
            point.tau[1] / dt * dNdx(a, 1) * N[b] * wJ;
        Ke(ip, jv) +=
            0.5 * point.tau[1] * dNdx(a, 1) * advection_b * wJ;

        Ke(ip, jp) +=
            point.tau[1] / rho * (dNdx(a, 0) * dNdx(b, 0) + dNdx(a, 1) * dNdx(b, 1)) * wJ;
      }
    }
  }
}

StabilizationRHS::StabilizationRHS(const std::vector<QPState>& state)
  : state_(state)
{
}

void StabilizationRHS::assemble(const ElementValues& ev,
                                Vector&              Fe) const
{
  const index_type num_shapes = ev.numDofs();

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    const auto& point = state_[static_cast<std::size_t>(q)];
    const auto  dNdx  = ev.dNdx(q);
    const auto  wJ    = ev.JxW(q);

    for (index_type a = 0; a < num_shapes; ++a)
    {
      const index_type iu = dim * a;
      const index_type iv = iu + 1;
      const index_type ip = dim * num_shapes + a;

      const real_type advection_a = advectionDerivative(ev, point, q, a);

      Fe[iu] +=
          point.tau[0] * rho / dt * advection_a * point.u[0] * wJ;
      Fe[iu] -=
          0.5 * point.tau[0] * rho * advection_a * point.grad_u_adv[0] * wJ;
      Fe[iv] +=
          point.tau[0] * rho / dt * advection_a * point.u[1] * wJ;
      Fe[iv] -=
          0.5 * point.tau[0] * rho * advection_a * point.grad_u_adv[1] * wJ;

      Fe[ip] +=
          point.tau[1] / dt * (dNdx(a, 0) * point.u[0] + dNdx(a, 1) * point.u[1]) * wJ;
      Fe[ip] -=
          0.5 * point.tau[1] * (dNdx(a, 0) * point.grad_u_adv[0] + dNdx(a, 1) * point.grad_u_adv[1]) * wJ;
    }
  }
}

} // namespace refem
