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

using namespace femx::assembly;
using namespace femx::system;

std::array<Real, 3> velAtNode(const Vector<Real>&   x,
                              const MixedFieldView& u_dof,
                              Index                 in)
{
  std::array<Real, 3> u{};
  for (Index d = 0; d < u_dof.numComponents(); ++d)
  {
    u[d] = x[u_dof.globalDof(in, d)];
  }
  return u;
}

void evalVel(const ElementValues& ev,
             const MixedFESpace&  space,
             Index                ic,
             Index                iq,
             const Vector<Real>&  x,
             std::array<Real, 3>& u)
{
  u              = {};
  const Index nd = ev.dim();

  const auto   N     = ev.N(iq);
  const Index* nodes = space.mesh().cellNodeIds(ic);
  const auto   u_dof = space.field(0);

  for (Index in = 0; in < ev.numNodes(); ++in)
  {
    const auto un = velAtNode(x, u_dof, nodes[in]);
    for (Index c = 0; c < nd; ++c)
    {
      u[c] += N[in] * un[c];
    }
  }
}

void evalVelGrad(const ElementValues& ev,
                 const MixedFESpace&  space,
                 Index                ic,
                 Index                iq,
                 const Vector<Real>&  x,
                 Real                 dudx[3][3])
{
  const Index nd = ev.dim();
  for (Index c = 0; c < 3; ++c)
  {
    for (Index d = 0; d < 3; ++d)
    {
      dudx[c][d] = 0.0;
    }
  }

  const auto   dNdx  = ev.dNdx(iq);
  const Index* nodes = space.mesh().cellNodeIds(ic);
  const auto   u_dof = space.field(0);

  for (Index in = 0; in < ev.numNodes(); ++in)
  {
    const auto un = velAtNode(x, u_dof, nodes[in]);
    for (Index c = 0; c < nd; ++c)
    {
      for (Index d = 0; d < nd; ++d)
      {
        dudx[c][d] += dNdx(in, d) * un[c];
      }
    }
  }
}

Real advectiveDerivative(const Real                 grad[3][3],
                         const std::array<Real, 3>& u_adv,
                         Index                      comp,
                         Index                      nd)
{
  Real value = 0.0;
  for (Index d = 0; d < nd; ++d)
  {
    value += grad[comp][d] * u_adv[d];
  }
  return value;
}

Real elemLength(const ElementValues&       ev,
                Index                      iq,
                const std::array<Real, 3>& u)
{
  const Index nd   = ev.dim();
  Real        mag2 = 0.0;
  for (Index d = 0; d < nd; ++d)
  {
    mag2 += u[d] * u[d];
  }

  const Real          mag = std::sqrt(mag2);
  std::array<Real, 3> dir{};
  if (mag > 1.0e-10)
  {
    for (Index d = 0; d < nd; ++d)
    {
      dir[d] = u[d] / mag;
    }
  }
  else
  {
    const Real value = 1.0 / std::sqrt(static_cast<Real>(nd));
    for (Index d = 0; d < nd; ++d)
    {
      dir[d] = value;
    }
  }

  const auto dNdx = ev.dNdx(iq);
  Real       sum  = 0.0;
  for (Index in = 0; in < ev.numNodes(); ++in)
  {
    Real grad_dir = 0.0;
    for (Index d = 0; d < nd; ++d)
    {
      grad_dir += dir[d] * dNdx(in, d);
    }
    sum += std::abs(grad_dir);
  }

  if (sum > 1.0e-14)
  {
    return 2.0 / sum;
  }
  return 0.0;
}

std::array<Real, 3> stabilization(
    const std::array<Real, 3>& u,
    const FluidParams&         fluid,
    Real                       dt,
    Real                       h,
    Index                      nd)
{
  const Real nu       = fluid.mu / fluid.rho;
  Real       vel_mag2 = 0.0;
  for (Index d = 0; d < nd; ++d)
  {
    vel_mag2 += u[d] * u[d];
  }

  const Real vel_mag = std::sqrt(vel_mag2);
  const Real term1   = std::pow(2.0 / dt, 2);
  Real       term2   = 0.0;
  Real       term3   = 0.0;
  if (h > 0.0)
  {
    term2 = std::pow(2.0 * vel_mag / h, 2);
    term3 = std::pow(4.0 * nu / (h * h), 2);
  }

  std::array<Real, 3> values{};
  values.fill(1.0 / std::sqrt(term1 + term2 + term3));
  return values;
}

void updateElemState(std::vector<QPState>& qps,
                     const ElementValues&  ev,
                     const MixedFESpace&   space,
                     Index                 ic,
                     const Vector<Real>&   x,
                     const Vector<Real>&   xp,
                     bool                  initial,
                     const FluidParams&    fluid,
                     Real                  dt,
                     Real&                 max_cfl)
{
  qps.resize(ev.numQuadraturePoints());

  auto qp_it = qps.begin();
  for (Index iq = 0; iq < ev.numQuadraturePoints(); ++iq, ++qp_it)
  {
    QPState& qp = *qp_it;

    const Index         nd = ev.dim();
    std::array<Real, 3> u_prev{};
    evalVel(ev, space, ic, iq, x, qp.u);
    evalVel(ev, space, ic, iq, xp, u_prev);
    evalVelGrad(ev, space, ic, iq, x, qp.grad_u);

    for (Index d = 0; d < nd; ++d)
    {
      qp.u_adv[d] = qp.u[d];
      if (!initial)
      {
        qp.u_adv[d] = 1.5 * qp.u[d] - 0.5 * u_prev[d];
      }
    }

    const Real h = elemLength(ev, iq, qp.u);
    if (h > 0.0)
    {
      Real vel_mag2 = 0.0;
      for (Index d = 0; d < nd; ++d)
      {
        vel_mag2 += qp.u[d] * qp.u[d];
      }
      max_cfl = std::max(max_cfl, std::sqrt(vel_mag2) * dt / h);
    }

    qp.tau = stabilization(qp.u, fluid, dt, h, nd);
    for (Index c = 0; c < nd; ++c)
    {
      qp.u_adv_grad_u[c] =
          advectiveDerivative(qp.grad_u, qp.u_adv, c, nd);
    }
  }
}

void assembleElemSystem(const MixedFESpace&   space,
                        Index                 ic,
                        ElementValues&        ev,
                        std::vector<QPState>& qps,
                        const Vector<Real>&   x,
                        const Vector<Real>&   xp,
                        bool                  initial,
                        const FluidParams&    fluid,
                        Real                  dt,
                        DenseMatrix&          Ke,
                        Vector<Real>&         Fe,
                        Real&                 max_cfl)
{
  ev.reinit(space.mesh().cell(ic));
  updateElemState(qps,
                  ev,
                  space,
                  ic,
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
                            Index               ic,
                            const DenseMatrix&  Ke,
                            const Vector<Real>& Fe,
                            const Vector<Real>& x_next,
                            Vector<Real>&       Re)
{
  const Index num_dofs = space.numDofsPerElem();
  if (Ke.rows() != num_dofs || Ke.cols() != num_dofs || Fe.size() != num_dofs)
  {
    throw std::runtime_error("Element system size does not match mixed space");
  }

  if (Re.size() != num_dofs)
  {
    Re.resize(num_dofs);
  }
  else
  {
    Re.setZero();
  }

  Vector<Index> dofs;
  space.elemDofs(ic, dofs);
  const Index* dof = dofs.data();

  for (Index i = 0; i < num_dofs; ++i)
  {
    Real value = -Fe[i];
    for (Index j = 0; j < num_dofs; ++j)
    {
      value += Ke(i, j) * x_next[dof[j]];
    }
    Re[i] = value;
  }
}

void assembleElemResidual(const MixedFESpace&   space,
                          Index                 ic,
                          ElementValues&        ev,
                          std::vector<QPState>& qps,
                          const Vector<Real>&   x_next,
                          const Vector<Real>&   x,
                          const Vector<Real>&   xp,
                          bool                  initial,
                          const FluidParams&    fluid,
                          Real                  dt,
                          Vector<Real>&         Re,
                          Real&                 max_cfl)
{
  DenseMatrix  Ke(space.numDofsPerElem(), space.numDofsPerElem());
  Vector<Real> Fe(space.numDofsPerElem());

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
  elemResidualFromSystem(space, ic, Ke, Fe, x_next, Re);
}

void assembleSystem(const MixedFESpace& space,
                    const Vector<Real>& x,
                    const Vector<Real>& xp,
                    bool                initial,
                    const FluidParams&  fluid,
                    Real                dt,
                    SparseSystemMatrix& A,
                    Vector<Real>&       b,
                    AssemblyStats&      stats)
{
  const auto& elem = space.field(0).space().finiteElement();
  const auto  quad = GaussQuadrature::make(elem.referenceElement(), 2);

  SystemAssembler initializer(space);
  initializer.initMat(A);
  initializer.initVec(b);
  Real max_cfl = 0.0;

#pragma omp parallel reduction(max : max_cfl)
  {
    ElementValues        ev(elem, quad);
    std::vector<QPState> qps;
    SystemAssembler      assembler(space, AssemblyMode::Atomic);

    DenseMatrix  Ke(space.numDofsPerElem(), space.numDofsPerElem());
    Vector<Real> Fe(space.numDofsPerElem());

#pragma omp for
    for (Index ic = 0; ic < space.mesh().numElems(); ++ic)
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
