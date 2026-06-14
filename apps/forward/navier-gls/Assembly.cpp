#include "Assembly.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "Components.hpp"
#include <femx/assembly/SystemAssembler.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>

namespace femx
{

std::array<real_type, 3> velAtNode(const Vector&         x,
                                   const MixedFieldView& u_dof,
                                   index_type            node)
{
  std::array<real_type, 3> u{};
  for (index_type d = 0; d < u_dof.numComponents(); ++d)
  {
    u[static_cast<std::size_t>(d)] = x[u_dof.globalDof(node, d)];
  }
  return u;
}

void evalVel(const ElementValues&      ev,
             const MixedFESpace&       space,
             index_type                cell,
             index_type                q,
             const Vector&             x,
             std::array<real_type, 3>& u)
{
  u                   = {};
  const index_type nd = ev.dim();

  const auto        N     = ev.N(q);
  const index_type* nodes = space.mesh().cellNodeIds(cell);
  const auto        u_dof = space.field(0);

  for (index_type in = 0; in < ev.numNodes(); ++in)
  {
    const auto un = velAtNode(x, u_dof, nodes[in]);
    for (index_type c = 0; c < nd; ++c)
    {
      u[c] += N[in] * un[c];
    }
  }
}

void evalVelGrad(const ElementValues& ev,
                 const MixedFESpace&  space,
                 index_type           cell,
                 index_type           q,
                 const Vector&        x,
                 real_type            dudx[3][3])
{
  const index_type nd = ev.dim();
  for (index_type c = 0; c < 3; ++c)
  {
    for (index_type d = 0; d < 3; ++d)
    {
      dudx[c][d] = 0.0;
    }
  }

  const auto        dNdx  = ev.dNdx(q);
  const index_type* nodes = space.mesh().cellNodeIds(cell);
  const auto        u_dof = space.field(0);

  for (index_type in = 0; in < ev.numNodes(); ++in)
  {
    const auto un = velAtNode(x, u_dof, nodes[in]);
    for (index_type c = 0; c < nd; ++c)
    {
      for (index_type d = 0; d < nd; ++d)
      {
        dudx[c][d] += dNdx(in, d) * un[c];
      }
    }
  }
}

real_type advectiveDerivative(const real_type                 grad[3][3],
                              const std::array<real_type, 3>& u_adv,
                              index_type                      comp,
                              index_type                      nd)
{
  real_type value = 0.0;
  for (index_type d = 0; d < nd; ++d)
  {
    value += grad[comp][d] * u_adv[d];
  }
  return value;
}

real_type elemLength(const ElementValues&            ev,
                     index_type                      q,
                     const std::array<real_type, 3>& u)
{
  const index_type nd   = ev.dim();
  real_type        mag2 = 0.0;
  for (index_type d = 0; d < nd; ++d)
  {
    mag2 += u[d] * u[d];
  }

  const real_type          mag = std::sqrt(mag2);
  std::array<real_type, 3> dir{};
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

  if (sum > 1.0e-14)
  {
    return 2.0 / sum;
  }
  return 0.0;
}

std::array<real_type, 3> stabilization(
    const std::array<real_type, 3>& u,
    const FluidParams&              fluid,
    real_type                       dt,
    real_type                       h,
    index_type                      nd)
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

  std::array<real_type, 3> values{};
  values.fill(1.0 / std::sqrt(term1 + term2 + term3));
  return values;
}

void updateElemState(std::vector<QPState>& qps,
                     const ElementValues&  ev,
                     const MixedFESpace&   space,
                     index_type            cell,
                     const Vector&         x,
                     const Vector&         xp,
                     bool                  initial,
                     const FluidParams&    fluid,
                     real_type             dt,
                     real_type&            max_cfl)
{
  qps.resize(static_cast<std::size_t>(ev.numQuadraturePoints()));

  for (index_type q = 0; q < ev.numQuadraturePoints(); ++q)
  {
    auto& qp = qps[static_cast<std::size_t>(q)];

    const index_type         nd = ev.dim();
    std::array<real_type, 3> u_prev{};
    evalVel(ev, space, cell, q, x, qp.u);
    evalVel(ev, space, cell, q, xp, u_prev);
    evalVelGrad(ev, space, cell, q, x, qp.grad_u);

    for (index_type d = 0; d < nd; ++d)
    {
      qp.u_adv[d] = qp.u[d];
      if (!initial)
      {
        qp.u_adv[d] = 1.5 * qp.u[d] - 0.5 * u_prev[d];
      }
    }

    const real_type h = elemLength(ev, q, qp.u);
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

void assembleElemSystem(const MixedFESpace&   space,
                        index_type            cell,
                        ElementValues&        ev,
                        std::vector<QPState>& qps,
                        const Vector&         x,
                        const Vector&         xp,
                        bool                  initial,
                        const FluidParams&    fluid,
                        real_type             dt,
                        DenseMatrix&          Ke,
                        Vector&               Fe,
                        real_type&            max_cfl)
{
  ev.reinit(space.mesh().cell(cell));
  updateElemState(qps,
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

  assembleMassLHS(ev, fluid, dt, Ke);
  assembleAdvectionLHS(ev, qps, fluid, Ke);
  assembleDiffusionLHS(ev, fluid, Ke);
  assemblePreVelCouplingLHS(ev, Ke);
  assembleStabilizationLHS(ev, qps, fluid, dt, Ke);

  assembleMassRHS(ev, qps, fluid, dt, Fe);
  assembleAdvectionRHS(ev, qps, fluid, Fe);
  assembleDiffusionRHS(ev, qps, fluid, Fe);
  assembleStabilizationRHS(ev, qps, fluid, dt, Fe);
}

void elemResidualFromSystem(const MixedFESpace& space,
                            index_type          cell,
                            const DenseMatrix&  Ke,
                            const Vector&       Fe,
                            const Vector&       x_next,
                            Vector&             Re)
{
  const index_type ndofs = space.numDofsPerElem();
  if (Ke.rows() != ndofs || Ke.cols() != ndofs || Fe.size() != ndofs)
  {
    throw std::runtime_error("Element system size does not match mixed space");
  }

  if (Re.size() != ndofs)
  {
    Re.resize(ndofs);
  }
  else
  {
    Re.setZero();
  }

  std::vector<index_type> dofs;
  space.elemDofs(cell, dofs);

  for (index_type i = 0; i < ndofs; ++i)
  {
    real_type value = -Fe[i];
    for (index_type j = 0; j < ndofs; ++j)
    {
      value += Ke(i, j) * x_next[dofs[static_cast<std::size_t>(j)]];
    }
    Re[i] = value;
  }
}

void assembleElemResidual(const MixedFESpace&   space,
                          index_type            cell,
                          ElementValues&        ev,
                          std::vector<QPState>& qps,
                          const Vector&         x_next,
                          const Vector&         x,
                          const Vector&         xp,
                          bool                  initial,
                          const FluidParams&    fluid,
                          real_type             dt,
                          Vector&               Re,
                          real_type&            max_cfl)
{
  DenseMatrix Ke(space.numDofsPerElem(), space.numDofsPerElem());
  Vector      Fe(space.numDofsPerElem());

  assembleElemSystem(space,
                     cell,
                     ev,
                     qps,
                     x,
                     xp,
                     initial,
                     fluid,
                     dt,
                     Ke,
                     Fe,
                     max_cfl);
  elemResidualFromSystem(space, cell, Ke, Fe, x_next, Re);
}

void assembleSystem(const MixedFESpace&         space,
                    const Vector&               x,
                    const Vector&               xp,
                    bool                        initial,
                    const FluidParams&          fluid,
                    real_type                   dt,
                    system::SparseSystemMatrix& A,
                    Vector&                     b,
                    AssemblyStats&              stats)
{
  const auto& elem = space.field(0).space().finiteElement();
  const auto  quad =
      GaussQuadrature::make(elem.referenceElement(), 2);
  const index_type nq = quad.size();

  assembly::SystemAssembler initializer(space);
  initializer.initMat(A);
  initializer.initVec(b);
  real_type max_cfl = 0.0;

#pragma omp parallel reduction(max : max_cfl)
  {
    ElementValues             ev(elem, quad);
    std::vector<QPState>      qps(static_cast<std::size_t>(nq));
    assembly::SystemAssembler assembler(
        space, assembly::SystemAssembler::AssemblyMode::Atomic);
    DenseMatrix Ke(space.numDofsPerElem(), space.numDofsPerElem());
    Vector      Fe(space.numDofsPerElem());

#pragma omp for
    for (index_type ic = 0; ic < space.mesh().numElems(); ++ic)
    {
      assembleElemSystem(space,
                         ic,
                         ev,
                         qps,
                         x,
                         xp,
                         initial,
                         fluid,
                         dt,
                         Ke,
                         Fe,
                         max_cfl);

      assembler.addMat(ic, Ke, A);
      assembler.addVec(ic, Fe, b);
    }
  }
  A.finalize();
  stats.max_cfl = max_cfl;
}

} // namespace femx
