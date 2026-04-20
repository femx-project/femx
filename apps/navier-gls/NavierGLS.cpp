#include "NavierGLS.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

#include "Components.hpp"
#include <refem/bc/DirichletCondition.hpp>
#include <refem/fe/BlockFESpace.hpp>
#include <refem/fe/ElementValues.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/fe/GaussQuadrature.hpp>
#include <refem/forms/integrators/DomainBilinearIntegrator.hpp>
#include <refem/forms/integrators/DomainLinearIntegrator.hpp>
#include <refem/linalg/DenseMatrix.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/mesh/Mesh.hpp>

namespace refem
{

bool near(real_type a, real_type b)
{
  return std::abs(a - b) <= 1.0e-12;
}

std::array<real_type, 2> velocityAtNode(const Vector&         x,
                                        const BlockFieldView& u_dof,
                                        index_type            in)
{
  const index_type i = u_dof.globalDof(in, 0);
  const index_type j = u_dof.globalDof(in, 1);
  return {x[i], x[j]};
}

void evaluateVelocity(const ElementValues&      ev,
                      const BlockFESpace&       space,
                      index_type                ic,
                      index_type                q,
                      const Vector&             x,
                      std::array<real_type, 2>& u,
                      real_type                 dudx[2][2])
{
  u = {0.0, 0.0};
  for (index_type c = 0; c < dim; ++c)
  {
    for (index_type d = 0; d < dim; ++d)
    {
      dudx[c][d] = 0.0;
    }
  }

  const auto        N     = ev.N(q);
  const auto        dNdx  = ev.dNdx(q);
  const index_type* nodes = space.mesh().cellNodeIds(ic);
  const auto        u_dof = space.field(0);

  for (index_type in = 0; in < ev.numNodes(); ++in)
  {
    const auto un = velocityAtNode(x, u_dof, nodes[in]);
    for (index_type c = 0; c < dim; ++c)
    {
      u[c] += N[in] * un[c];
      for (index_type d = 0; d < dim; ++d)
      {
        dudx[c][d] += dNdx(in, d) * un[c];
      }
    }
  }
}

real_type advectiveDerivative(const real_type                 grad[2][2],
                              const std::array<real_type, 2>& u_adv,
                              index_type                      component)
{
  return grad[component][0] * u_adv[0] + grad[component][1] * u_adv[1];
}

real_type elemLength(const ElementValues&            ev,
                     index_type                      q,
                     const std::array<real_type, 2>& u)
{
  const real_type          mag = std::sqrt(u[0] * u[0] + u[1] * u[1]);
  std::array<real_type, 2> dir{};

  if (mag > 1.0e-10)
  {
    dir = {u[0] / mag, u[1] / mag};
  }
  else
  {
    const real_type tmp = 1.0 / std::sqrt(2.0);
    dir                 = {tmp, tmp};
  }

  const auto dNdx = ev.dNdx(q);
  real_type  sum  = 0.0;
  for (index_type a = 0; a < ev.numNodes(); ++a)
  {
    sum += std::abs(dir[0] * dNdx(a, 0) + dir[1] * dNdx(a, 1));
  }

  if (sum > 1.0e-14)
  {
    return 2.0 / sum;
  }
  return 0.0;
}

std::array<real_type, 2> stabilization(const std::array<real_type, 2>& u,
                                       real_type                       h)
{
  const real_type nu      = mu / rho;
  const real_type vel_mag = std::sqrt(u[0] * u[0] + u[1] * u[1]);
  const real_type term1   = std::pow(2.0 / dt, 2);
  real_type       term2   = 0.0;
  real_type       term3   = 0.0;
  if (h > 0.0)
  {
    term2 = std::pow(2.0 * vel_mag / h, 2);
    term3 = std::pow(4.0 * nu / (h * h), 2);
  }
  const real_type tau = 1.0 / std::sqrt(term1 + term2 + term3);

  return {tau, tau};
}

std::vector<QPState> makeCellState(const ElementValues& ev,
                                   const BlockFESpace&  space,
                                   index_type           ic,
                                   const Vector&        x,
                                   const Vector&        xp,
                                   bool                 initial,
                                   real_type&           max_cfl)
{
  const index_type nq = ev.numQuadraturePoints();

  std::vector<QPState> state(static_cast<std::size_t>(nq));

  for (index_type q = 0; q < nq; ++q)
  {
    auto& point = state[static_cast<std::size_t>(q)];

    std::array<real_type, 2> u_prev{};
    real_type                grad_unused[2][2]{};
    evaluateVelocity(ev, space, ic, q, x, point.u, point.grad);
    evaluateVelocity(ev, space, ic, q, xp, u_prev, grad_unused);

    if (initial)
    {
      point.u_adv = point.u;
    }
    else
    {
      point.u_adv = {1.5 * point.u[0] - 0.5 * u_prev[0],
                     1.5 * point.u[1] - 0.5 * u_prev[1]};
    }

    const real_type h = elemLength(ev, q, point.u);
    if (h > 0.0)
    {
      const real_type cfl =
          std::sqrt(point.u[0] * point.u[0] + point.u[1] * point.u[1]) * dt / h;
      max_cfl = std::max(max_cfl, cfl);
    }

    point.tau = stabilization(point.u, h);
    for (index_type c = 0; c < dim; ++c)
    {
      point.grad_u_adv[c] =
          advectiveDerivative(point.grad, point.u_adv, c);
    }
  }

  return state;
}

void splitFields(const Vector&       x,
                 const BlockFESpace& space,
                 index_type          nodes,
                 Vector&             ux,
                 Vector&             uy,
                 Vector&             p)
{
  const auto u_dof = space.field(0);
  const auto p_dof = space.field(1);
  for (index_type in = 0; in < nodes; ++in)
  {
    ux[in] = x[u_dof.globalDof(in, 0)];
    uy[in] = x[u_dof.globalDof(in, 1)];
    p[in]  = x[p_dof.globalDof(in)];
  }
}

DirichletCondition cavityBoundary(const BlockFESpace& space)
{
  DirichletCondition bc;
  const Mesh&        mesh  = space.mesh();
  const auto         u_dof = space.field(0);
  const auto         p_dof = space.field(1);

  for (index_type in = 0; in < mesh.numNodes(); ++in)
  {
    const auto& coord = mesh.node(in);
    if (near(coord[0], 0.0) || near(coord[0], 1.0) || near(coord[1], 0.0) || near(coord[1], 1.0))
    {
      real_type value = 0.0;
      if (near(coord[1], 1.0))
      {
        value = lid;
      }
      bc.addDof(u_dof.globalDof(in, 0), value);
      bc.addDof(u_dof.globalDof(in, 1), 0.0);
    }
  }

  bc.addDof(p_dof.globalDof(0), 0.0);
  return bc;
}

void assembleSystem(const BlockFESpace& space,
                    const Vector&       x,
                    const Vector&       xp,
                    bool                initial,
                    SparseMatrix&       A,
                    Vector&             b,
                    real_type&          max_cfl)
{
  const auto& element = space.field(0).space().finiteElement();
  const auto  quadrature =
      GaussQuadrature::make(element.referenceElement(), 2);
  ElementValues ev(element, quadrature);

  A.setZero();
  b.setZero();
  max_cfl = 0.0;

  for (index_type ic = 0; ic < space.mesh().numElems(); ++ic)
  {
    ev.reinit(space.mesh().cells()[static_cast<std::size_t>(ic)]);
    std::vector<QPState> state = makeCellState(ev,
                                               space,
                                               ic,
                                               x,
                                               xp,
                                               initial,
                                               max_cfl);

    DenseMatrix Ke(space.numDofsPerElem(), space.numDofsPerElem());
    Vector      Fe(space.numDofsPerElem());
    Ke.setZero();
    Fe.setZero();

    std::vector<std::unique_ptr<DomainBilinearIntegrator>> lhs_terms;
    lhs_terms.push_back(std::make_unique<TransientLHS>());
    lhs_terms.push_back(std::make_unique<AdvectionLHS>(state));
    lhs_terms.push_back(std::make_unique<ViscousLHS>());
    lhs_terms.push_back(std::make_unique<PressureVelocityCouplingLHS>());
    lhs_terms.push_back(std::make_unique<StabilizationLHS>(state));

    std::vector<std::unique_ptr<DomainLinearIntegrator>> rhs_terms;
    rhs_terms.push_back(std::make_unique<TransientRHS>(state));
    rhs_terms.push_back(std::make_unique<AdvectionRHS>(state));
    rhs_terms.push_back(std::make_unique<ViscousRHS>(state));
    rhs_terms.push_back(std::make_unique<StabilizationRHS>(state));

    for (const auto& term : lhs_terms)
    {
      term->assemble(ev, Ke);
    }

    for (const auto& term : rhs_terms)
    {
      term->assemble(ev, Fe);
    }

    A.addLocalMatrix(ic, Ke);
    b.addLocalVector(space.elemDofs(ic), Fe);
  }
}

Snapshot makeSnapshot(const BlockFESpace& space,
                      const Vector&       x,
                      real_type           time)
{
  const Mesh& mesh = space.mesh();
  Snapshot    snapshot{time,
                       Vector(mesh.numNodes()),
                       Vector(mesh.numNodes()),
                       Vector(mesh.numNodes())};
  splitFields(x, space, mesh.numNodes(), snapshot.ux, snapshot.uy, snapshot.p);
  return snapshot;
}

} // namespace refem
