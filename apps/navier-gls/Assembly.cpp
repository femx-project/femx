#include "Assembly.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "Components.hpp"
#include <refem/fe/BlockFESpace.hpp>
#include <refem/fe/ElementValues.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/fe/GaussQuadrature.hpp>
#include <refem/linalg/DenseMatrix.hpp>
#include <refem/linalg/LocalAssembler.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/mesh/Mesh.hpp>

namespace refem
{

std::array<real_type, max_dim> velocityAtNode(const Vector&         x,
                                              const BlockFieldView& u_dof,
                                              index_type            node)
{
  std::array<real_type, max_dim> u{};
  for (index_type d = 0; d < u_dof.numComponents(); ++d)
  {
    u[static_cast<std::size_t>(d)] = x[u_dof.globalDof(node, d)];
  }
  return u;
}

void evaluateVelocity(const ElementValues&            ev,
                      const BlockFESpace&             space,
                      index_type                      cell,
                      index_type                      q,
                      const Vector&                   x,
                      std::array<real_type, max_dim>& u)
{
  u                   = {};
  const index_type nd = ev.dim();

  const auto        N     = ev.N(q);
  const index_type* nodes = space.mesh().cellNodeIds(cell);
  const auto        u_dof = space.field(0);

  for (index_type in = 0; in < ev.numNodes(); ++in)
  {
    const auto un = velocityAtNode(x, u_dof, nodes[in]);
    for (index_type c = 0; c < nd; ++c)
    {
      u[c] += N[in] * un[c];
    }
  }
}

void evaluateVelocityGradient(const ElementValues& ev,
                              const BlockFESpace&  space,
                              index_type           cell,
                              index_type           q,
                              const Vector&        x,
                              real_type            dudx[max_dim][max_dim])
{
  const index_type nd = ev.dim();
  for (index_type c = 0; c < max_dim; ++c)
  {
    for (index_type d = 0; d < max_dim; ++d)
    {
      dudx[c][d] = 0.0;
    }
  }

  const auto        dNdx  = ev.dNdx(q);
  const index_type* nodes = space.mesh().cellNodeIds(cell);
  const auto        u_dof = space.field(0);

  for (index_type in = 0; in < ev.numNodes(); ++in)
  {
    const auto un = velocityAtNode(x, u_dof, nodes[in]);
    for (index_type c = 0; c < nd; ++c)
    {
      for (index_type d = 0; d < nd; ++d)
      {
        dudx[c][d] += dNdx(in, d) * un[c];
      }
    }
  }
}

real_type advectiveDerivative(const real_type                       grad[max_dim][max_dim],
                              const std::array<real_type, max_dim>& u_adv,
                              index_type                            component,
                              index_type                            nd)
{
  real_type value = 0.0;
  for (index_type d = 0; d < nd; ++d)
  {
    value += grad[component][d] * u_adv[d];
  }
  return value;
}

real_type elementLength(const ElementValues&                  ev,
                        index_type                            q,
                        const std::array<real_type, max_dim>& u)
{
  const index_type nd   = ev.dim();
  real_type        mag2 = 0.0;
  for (index_type d = 0; d < nd; ++d)
  {
    mag2 += u[d] * u[d];
  }

  const real_type                mag = std::sqrt(mag2);
  std::array<real_type, max_dim> dir{};
  if (mag > 1.0e-10)
  {
    for (index_type d = 0; d < nd; ++d)
    {
      dir[d] = u[d] / mag;
    }
  }
  else
  {
    const real_type value = 1.0 / std::sqrt(static_cast<real_type>(nd));
    for (index_type d = 0; d < nd; ++d)
    {
      dir[d] = value;
    }
  }

  const auto dNdx = ev.dNdx(q);
  real_type  sum  = 0.0;
  for (index_type i = 0; i < ev.numNodes(); ++i)
  {
    real_type grad_dir = 0.0;
    for (index_type d = 0; d < nd; ++d)
    {
      grad_dir += dir[d] * dNdx(i, d);
    }
    sum += std::abs(grad_dir);
  }

  return sum > 1.0e-14 ? 2.0 / sum : 0.0;
}

std::array<real_type, max_dim> stabilization(
    const std::array<real_type, max_dim>& u,
    const FluidParams&                    fluid,
    real_type                             dt,
    real_type                             h,
    index_type                            nd)
{
  const real_type nu       = fluid.mu / fluid.rho;
  real_type       vel_mag2 = 0.0;
  for (index_type d = 0; d < nd; ++d)
  {
    vel_mag2 += u[d] * u[d];
  }

  const real_type vel_mag = std::sqrt(vel_mag2);
  const real_type term1   = std::pow(2.0 / dt, 2);
  real_type       term2   = 0.0;
  real_type       term3   = 0.0;
  if (h > 0.0)
  {
    term2 = std::pow(2.0 * vel_mag / h, 2);
    term3 = std::pow(4.0 * nu / (h * h), 2);
  }

  std::array<real_type, max_dim> values{};
  values.fill(1.0 / std::sqrt(term1 + term2 + term3));
  return values;
}

void updateCellState(std::vector<QPState>& qp_states,
                     const ElementValues&  ev,
                     const BlockFESpace&   space,
                     index_type            cell,
                     const Vector&         x,
                     const Vector&         xp,
                     bool                  initial,
                     const FluidParams&    fluid,
                     real_type             dt,
                     real_type&            max_cfl)
{
  qp_states.resize(static_cast<std::size_t>(ev.numQuadraturePoints()));

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    auto& qp = qp_states[static_cast<std::size_t>(q)];

    const index_type               nd = ev.dim();
    std::array<real_type, max_dim> u_prev{};
    evaluateVelocity(ev, space, cell, q, x, qp.u);
    evaluateVelocityGradient(ev, space, cell, q, x, qp.grad_u);
    evaluateVelocity(ev, space, cell, q, xp, u_prev);

    for (index_type d = 0; d < nd; ++d)
    {
      qp.u_adv[d] = initial ? qp.u[d] : 1.5 * qp.u[d] - 0.5 * u_prev[d];
    }

    const real_type h = elementLength(ev, q, qp.u);
    if (h > 0.0)
    {
      real_type vel_mag2 = 0.0;
      for (index_type d = 0; d < nd; ++d)
      {
        vel_mag2 += qp.u[d] * qp.u[d];
      }
      max_cfl = std::max(max_cfl, std::sqrt(vel_mag2) * dt / h);
    }

    qp.tau = stabilization(qp.u, fluid, dt, h, nd);
    for (index_type c = 0; c < nd; ++c)
    {
      qp.u_adv_grad_u[c] =
          advectiveDerivative(qp.grad_u, qp.u_adv, c, nd);
    }
  }
}

void assembleSystem(const BlockFESpace& space,
                    const Vector&       x,
                    const Vector&       xp,
                    bool                initial,
                    const FluidParams&  fluid,
                    real_type           dt,
                    SparseMatrix&       A,
                    Vector&             b,
                    AssemblyStats&      stats)
{
  const auto& element = space.field(0).space().finiteElement();
  const auto  quadrature =
      GaussQuadrature::make(element.referenceElement(), 2);
  const index_type nq = quadrature.size();

  A.setZero();
  b.setZero();
  real_type max_cfl = 0.0;

#pragma omp parallel reduction(max : max_cfl)
  {
    ElementValues        ev(element, quadrature);
    std::vector<QPState> qp_states(static_cast<std::size_t>(nq));
    LocalAssembler       assembler(space,
                             A.pattern(),
                             LocalAssembler::AssemblyPolicy::Atomic);
    DenseMatrix          Ke(space.numDofsPerElem(), space.numDofsPerElem());
    Vector               Fe(space.numDofsPerElem());

#pragma omp for
    for (index_type cell = 0; cell < space.mesh().numElems(); ++cell)
    {
      ev.reinit(space.mesh().cell(cell));
      updateCellState(qp_states,
                      ev,
                      space,
                      cell,
                      x,
                      xp,
                      initial,
                      fluid,
                      dt,
                      max_cfl);

      Ke.setZero();
      Fe.setZero();

      assembleTransientLHS(ev, fluid, dt, Ke);
      assembleAdvectionLHS(ev, qp_states, fluid, Ke);
      assembleViscousLHS(ev, fluid, Ke);
      assemblePressureVelocityCouplingLHS(ev, Ke);
      assembleStabilizationLHS(ev, qp_states, fluid, dt, Ke);

      assembleTransientRHS(ev, qp_states, fluid, dt, Fe);
      assembleAdvectionRHS(ev, qp_states, fluid, Fe);
      assembleViscousRHS(ev, qp_states, fluid, Fe);
      assembleStabilizationRHS(ev, qp_states, fluid, dt, Fe);

      assembler.addLocalMatrix(cell, Ke, A);
      assembler.addLocalVector(cell, Fe, b);
    }
  }
  stats.max_cfl = max_cfl;
}

} // namespace refem
