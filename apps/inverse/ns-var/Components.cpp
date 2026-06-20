#include "Components.hpp"

#include <cmath>

namespace femx
{

namespace
{

Index vel(Index node, Index comp, Index dim)
{
  return dim * node + comp;
}

Index pressure(Index node, Index num_shape, Index dim)
{
  return dim * num_shape + node;
}

Index num_dofs(Index num_shape, Index dim)
{
  return dim * num_shape + num_shape;
}

Real adv(const Real* dNdx, Index i, const Real u[3], Index dim)
{
  Real value = 0.0;
  for (Index d = 0; d < dim; ++d)
  {
    value += dNdx[i * dim + d] * u[d];
  }
  return value;
}

Real shapeDot(const Real* dNdx, Index i, Index j, Index dim)
{
  Real value = 0.0;
  for (Index d = 0; d < dim; ++d)
  {
    value += dNdx[i * dim + d] * dNdx[j * dim + d];
  }
  return value;
}

} // namespace

void evalQp(Index       num_shape,
            Index       dim,
            const Real* N,
            const Real* dNdx,
            const Real* x,
            Qp&         qp)
{
  qp = {};
  for (Index i = 0; i < num_shape; ++i)
  {
    for (Index c = 0; c < dim; ++c)
    {
      const Real x_c  = x[vel(i, c, dim)];
      qp.u[c]        += N[i] * x_c;
      for (Index d = 0; d < dim; ++d)
      {
        qp.grad[c][d] += dNdx[i * dim + d] * x_c;
      }
    }
  }

  for (Index c = 0; c < dim; ++c)
  {
    for (Index d = 0; d < dim; ++d)
    {
      qp.adv_grad[c] += qp.grad[c][d] * qp.u[d];
    }
  }
}

Real elemLength(Index num_shape, Index dim, const Real* dNdx, const Real u[3])
{
  Real mag2 = 0.0;
  for (Index d = 0; d < dim; ++d)
  {
    mag2 += u[d] * u[d];
  }

  const Real mag = std::sqrt(mag2);
  Real       dir[3]{};
  if (mag > 1.0e-10)
  {
    for (Index d = 0; d < dim; ++d)
    {
      dir[d] = u[d] / mag;
    }
  }
  else
  {
    const Real value = 1.0 / std::sqrt(static_cast<Real>(dim));
    for (Index d = 0; d < dim; ++d)
    {
      dir[d] = value;
    }
  }

  Real sum = 0.0;
  for (Index i = 0; i < num_shape; ++i)
  {
    Real grad_dir = 0.0;
    for (Index d = 0; d < dim; ++d)
    {
      grad_dir += dir[d] * dNdx[i * dim + d];
    }
    sum += grad_dir >= 0.0 ? grad_dir : -grad_dir;
  }

  return sum > 1.0e-14 ? 2.0 / sum : 0.0;
}

Real glsTau(const Real u[3], Real rho, Real mu, Real dt, Real h, Index dim)
{
  Real speed2 = 0.0;
  for (Index d = 0; d < dim; ++d)
  {
    speed2 += u[d] * u[d];
  }

  const Real speed = std::sqrt(speed2);
  const Real nu    = mu / rho;
  const Real time  = (2.0 / dt) * (2.0 / dt);
  Real       flow  = 0.0;
  Real       diff  = 0.0;
  if (h > 0.0)
  {
    flow = (2.0 * speed / h) * (2.0 * speed / h);
    diff = (4.0 * nu / (h * h)) * (4.0 * nu / (h * h));
  }
  return 1.0 / std::sqrt(time + flow + diff);
}

void updateElemState(Qp*         qps,
                     Index       num_qp,
                     Index       num_shape,
                     Index       dim,
                     const Real* N,
                     const Real* dNdx,
                     const Real* x,
                     Real        rho,
                     Real        mu,
                     Real        dt)
{
  for (Index iq = 0; iq < num_qp; ++iq)
  {
    const Real* Nq     = N + iq * num_shape;
    const Real* dNdx_q = dNdx + iq * num_shape * dim;

    Qp& qp = qps[iq];
    evalQp(num_shape, dim, Nq, dNdx_q, x, qp);
    const Real h = elemLength(num_shape, dim, dNdx_q, qp.u);
    qp.tau       = glsTau(qp.u, rho, mu, dt, h, dim);
  }
}

void assembleMassLHS(Index       num_shape,
                     Index       dim,
                     Real        rho,
                     Real        dt,
                     const Real* N,
                     Real        Jw,
                     Real*       Ke)
{
  const Index n = num_dofs(num_shape, dim);
  for (Index i = 0; i < num_shape; ++i)
  {
    for (Index j = 0; j < num_shape; ++j)
    {
      const Real value = rho / dt * N[i] * N[j] * Jw;
      for (Index d = 0; d < dim; ++d)
      {
        Ke[vel(i, d, dim) * n + vel(j, d, dim)] += value;
      }
    }
  }
}

void assembleAdvectionLHS(Index       num_shape,
                          Index       dim,
                          Real        rho,
                          const Real* N,
                          const Real* dNdx,
                          Real        Jw,
                          const Qp&   qp,
                          Real*       Ke)
{
  const Index n = num_dofs(num_shape, dim);
  for (Index i = 0; i < num_shape; ++i)
  {
    for (Index j = 0; j < num_shape; ++j)
    {
      const Real value = 0.5 * rho * N[i] * adv(dNdx, j, qp.u, dim) * Jw;
      for (Index d = 0; d < dim; ++d)
      {
        Ke[vel(i, d, dim) * n + vel(j, d, dim)] += value;
      }
    }
  }
}

void assembleDiffusionLHS(Index       num_shape,
                          Index       dim,
                          Real        mu,
                          const Real* dNdx,
                          Real        Jw,
                          Real*       Ke)
{
  const Index n = num_dofs(num_shape, dim);
  for (Index i = 0; i < num_shape; ++i)
  {
    for (Index j = 0; j < num_shape; ++j)
    {
      const Real value = 0.5 * mu * shapeDot(dNdx, i, j, dim) * Jw;
      for (Index d = 0; d < dim; ++d)
      {
        Ke[vel(i, d, dim) * n + vel(j, d, dim)] += value;
      }
    }
  }
}

void assemblePressureLHS(Index       num_shape,
                         Index       dim,
                         const Real* N,
                         const Real* dNdx,
                         Real        Jw,
                         Real*       Ke)
{
  const Index n = num_dofs(num_shape, dim);
  for (Index i = 0; i < num_shape; ++i)
  {
    const Index ip = pressure(i, num_shape, dim);
    for (Index j = 0; j < num_shape; ++j)
    {
      const Index jp = pressure(j, num_shape, dim);
      for (Index d = 0; d < dim; ++d)
      {
        Ke[vel(i, d, dim) * n + jp] -= dNdx[i * dim + d] * N[j] * Jw;
        Ke[ip * n + vel(j, d, dim)] += N[i] * dNdx[j * dim + d] * Jw;
      }
    }
  }
}

void assembleStabilizationLHS(Index       num_shape,
                              Index       dim,
                              Real        rho,
                              Real        dt,
                              const Real* N,
                              const Real* dNdx,
                              Real        Jw,
                              const Qp&   qp,
                              Real*       Ke)
{
  const Index n = num_dofs(num_shape, dim);
  for (Index i = 0; i < num_shape; ++i)
  {
    const Index ip    = pressure(i, num_shape, dim);
    const Real  dvidx = adv(dNdx, i, qp.u, dim);

    for (Index j = 0; j < num_shape; ++j)
    {
      const Index jp        = pressure(j, num_shape, dim);
      const Real  dvjdx     = adv(dNdx, j, qp.u, dim);
      const Real  shape_dot = shapeDot(dNdx, i, j, dim);

      for (Index d = 0; d < dim; ++d)
      {
        const Index iu   = vel(i, d, dim);
        const Index ju   = vel(j, d, dim);
        Ke[iu * n + ju] += qp.tau * rho / dt * dvidx * N[j] * Jw;
        Ke[iu * n + ju] += 0.5 * qp.tau * rho * dvidx * dvjdx * Jw;
        Ke[iu * n + jp] += qp.tau * dvidx * dNdx[j * dim + d] * Jw;
        Ke[ip * n + ju] += qp.tau / dt * dNdx[i * dim + d] * N[j] * Jw;
        Ke[ip * n + ju] += 0.5 * qp.tau * dNdx[i * dim + d] * dvjdx * Jw;
      }

      Ke[ip * n + jp] += qp.tau / rho * shape_dot * Jw;
    }
  }
}

void assembleMassRHS(Index       num_shape,
                     Index       dim,
                     Real        rho,
                     Real        dt,
                     const Real* N,
                     Real        Jw,
                     const Qp&   qp,
                     Real*       Fe)
{
  for (Index i = 0; i < num_shape; ++i)
  {
    for (Index d = 0; d < dim; ++d)
    {
      Fe[vel(i, d, dim)] += rho / dt * N[i] * qp.u[d] * Jw;
    }
  }
}

void assembleAdvectionRHS(Index       num_shape,
                          Index       dim,
                          Real        rho,
                          const Real* N,
                          Real        Jw,
                          const Qp&   qp,
                          Real*       Fe)
{
  for (Index i = 0; i < num_shape; ++i)
  {
    for (Index d = 0; d < dim; ++d)
    {
      Fe[vel(i, d, dim)] -= 0.5 * rho * N[i] * qp.adv_grad[d] * Jw;
    }
  }
}

void assembleDiffusionRHS(Index       num_shape,
                          Index       dim,
                          Real        mu,
                          const Real* dNdx,
                          Real        Jw,
                          const Qp&   qp,
                          Real*       Fe)
{
  for (Index i = 0; i < num_shape; ++i)
  {
    for (Index c = 0; c < dim; ++c)
    {
      Real value = 0.0;
      for (Index d = 0; d < dim; ++d)
      {
        value += dNdx[i * dim + d] * qp.grad[c][d];
      }
      Fe[vel(i, c, dim)] -= 0.5 * mu * value * Jw;
    }
  }
}

void assembleStabilizationRHS(Index       num_shape,
                              Index       dim,
                              Real        rho,
                              Real        dt,
                              const Real* dNdx,
                              Real        Jw,
                              const Qp&   qp,
                              Real*       Fe)
{
  for (Index i = 0; i < num_shape; ++i)
  {
    const Index ip             = pressure(i, num_shape, dim);
    const Real  dvidx          = adv(dNdx, i, qp.u, dim);
    Real        div_u          = 0.0;
    Real        div_adv_grad_u = 0.0;

    for (Index d = 0; d < dim; ++d)
    {
      const Index iu  = vel(i, d, dim);
      Fe[iu]         += qp.tau * rho / dt * dvidx * qp.u[d] * Jw;
      Fe[iu]         -= 0.5 * qp.tau * rho * dvidx * qp.adv_grad[d] * Jw;
      div_u          += dNdx[i * dim + d] * qp.u[d];
      div_adv_grad_u += dNdx[i * dim + d] * qp.adv_grad[d];
    }

    Fe[ip] += qp.tau / dt * div_u * Jw;
    Fe[ip] -= 0.5 * qp.tau * div_adv_grad_u * Jw;
  }
}

} // namespace femx
