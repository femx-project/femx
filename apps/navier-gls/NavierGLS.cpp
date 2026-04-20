#include "NavierGLS.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

#include "Components.hpp"
#include <refem/bc/DirichletCondition.hpp>
#include <refem/fe/BlockFESpace.hpp>
#include <refem/fe/ElementValues.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/fe/GaussQuadrature.hpp>
#include <refem/linalg/DenseMatrix.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/mesh/Mesh.hpp>

namespace refem
{

bool near(real_type a, real_type b)
{
  return std::abs(a - b) <= 1.0e-12;
}

std::array<real_type, dim> velocityAtNode(const Vector&         x,
                                          const BlockFieldView& u_dof,
                                          index_type            in)
{
  std::array<real_type, dim> u{};
  for (index_type d = 0; d < u_dof.numComponents(); ++d)
  {
    u[static_cast<std::size_t>(d)] = x[u_dof.globalDof(in, d)];
  }
  return u;
}

void evaluateVelocity(const ElementValues&        ev,
                      const BlockFESpace&         space,
                      index_type                  ic,
                      index_type                  q,
                      const Vector&               x,
                      std::array<real_type, dim>& u,
                      real_type                   dudx[dim][dim])
{
  u                   = {};
  const index_type nd = ev.dim();
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
    for (index_type c = 0; c < nd; ++c)
    {
      u[c] += N[in] * un[c];
      for (index_type d = 0; d < nd; ++d)
      {
        dudx[c][d] += dNdx(in, d) * un[c];
      }
    }
  }
}

real_type advectiveDerivative(const real_type                   grad[dim][dim],
                              const std::array<real_type, dim>& u_adv,
                              index_type                        component,
                              index_type                        nd)
{
  real_type value = 0.0;
  for (index_type d = 0; d < nd; ++d)
  {
    value += grad[component][d] * u_adv[d];
  }
  return value;
}

real_type elemLength(const ElementValues&              ev,
                     index_type                        q,
                     const std::array<real_type, dim>& u)
{
  const index_type nd   = ev.dim();
  real_type        mag2 = 0.0;
  for (index_type d = 0; d < nd; ++d)
  {
    mag2 += u[d] * u[d];
  }
  const real_type            mag = std::sqrt(mag2);
  std::array<real_type, dim> dir{};

  if (mag > 1.0e-10)
  {
    for (index_type d = 0; d < nd; ++d)
    {
      dir[d] = u[d] / mag;
    }
  }
  else
  {
    const real_type tmp = 1.0 / std::sqrt(static_cast<real_type>(nd));
    for (index_type d = 0; d < nd; ++d)
    {
      dir[d] = tmp;
    }
  }

  const auto dNdx = ev.dNdx(q);
  real_type  sum  = 0.0;
  for (index_type i = 0; i < ev.numNodes(); ++i)
  {
    real_type directional_grad = 0.0;
    for (index_type d = 0; d < nd; ++d)
    {
      directional_grad += dir[d] * dNdx(i, d);
    }
    sum += std::abs(directional_grad);
  }

  if (sum > 1.0e-14)
  {
    return 2.0 / sum;
  }
  return 0.0;
}

std::array<real_type, dim> stabilization(const std::array<real_type, dim>& u,
                                         real_type                         h,
                                         index_type                        nd)
{
  const real_type nu       = mu / rho;
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
  const real_type tau = 1.0 / std::sqrt(term1 + term2 + term3);

  std::array<real_type, dim> values{};
  values.fill(tau);
  return values;
}

void updateCellState(std::vector<QPState>& qp_states,
                     const ElementValues&  ev,
                     const BlockFESpace&   space,
                     index_type            ic,
                     const Vector&         x,
                     const Vector&         xp,
                     bool                  initial,
                     real_type&            observed_max_cfl)
{
  const index_type nq = ev.numQuadraturePoints();

  qp_states.resize(static_cast<std::size_t>(nq));

  for (index_type q = 0; q < nq; ++q)
  {
    auto& qp = qp_states[static_cast<std::size_t>(q)];

    const index_type           nd = ev.dim();
    std::array<real_type, dim> u_prev{};
    real_type                  grad_unused[dim][dim]{};
    evaluateVelocity(ev, space, ic, q, x, qp.u, qp.grad_u);
    evaluateVelocity(ev, space, ic, q, xp, u_prev, grad_unused);

    if (initial)
    {
      qp.u_adv = qp.u;
    }
    else
    {
      for (index_type d = 0; d < nd; ++d)
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
      const real_type cfl = std::sqrt(vel_mag2) * dt / h;
      observed_max_cfl    = std::max(observed_max_cfl, cfl);
    }

    qp.tau = stabilization(qp.u, h, nd);
    for (index_type c = 0; c < nd; ++c)
    {
      qp.u_adv_grad_u[c] =
          advectiveDerivative(qp.grad_u, qp.u_adv, c, nd);
    }
  }

}

void splitFields(const Vector&       x,
                 const BlockFESpace& space,
                 index_type          nodes,
                 Vector&             ux,
                 Vector&             uy,
                 Vector&             uz,
                 Vector&             p)
{
  const auto       u_dof = space.field(0);
  const auto       p_dof = space.field(1);
  const index_type nd    = u_dof.numComponents();
  for (index_type in = 0; in < nodes; ++in)
  {
    ux[in] = x[u_dof.globalDof(in, 0)];
    uy[in] = nd > 1 ? x[u_dof.globalDof(in, 1)] : 0.0;
    uz[in] = nd > 2 ? x[u_dof.globalDof(in, 2)] : 0.0;
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

std::set<index_type> boundaryNodes(const Mesh&        mesh,
                                   const std::string& physical_name)
{
  std::set<index_type> nodes;
  for (const auto& facet : mesh.boundaryFacets(physical_name))
  {
    for (index_type node : facet.node_ids)
    {
      nodes.insert(node);
    }
  }
  return nodes;
}

std::set<index_type> boundaryNodes(const Mesh& mesh,
                                   index_type  physical_tag)
{
  std::set<index_type> nodes;
  for (const auto& facet : mesh.boundaryFacets())
  {
    if (facet.physical_tag == physical_tag)
    {
      for (index_type node : facet.node_ids)
      {
        nodes.insert(node);
      }
    }
  }
  return nodes;
}

DirichletCondition configuredBoundary(const BlockFESpace& space,
                                      bool&               has_pressure_bc)
{
  const Mesh& mesh  = space.mesh();
  const auto  u_dof = space.field(0);
  const auto  p_dof = space.field(1);

  DirichletCondition bc;
  has_pressure_bc = false;

  for (const auto& condition : bcs)
  {
    const auto nodes = boundaryNodes(mesh, condition.physical_tag);
    if (nodes.empty())
    {
      throw std::runtime_error(
          "No boundary facets found for physical tag " + std::to_string(condition.physical_tag));
    }

    for (index_type node : nodes)
    {
      if (condition.has_ux)
      {
        bc.addDof(u_dof.globalDof(node, 0), condition.ux);
      }
      if (condition.has_uy)
      {
        bc.addDof(u_dof.globalDof(node, 1), condition.uy);
      }
      if (condition.has_uz)
      {
        if (u_dof.numComponents() < 3)
        {
          throw std::runtime_error("uz boundary condition requires a 3D mesh");
        }
        bc.addDof(u_dof.globalDof(node, 2), condition.uz);
      }
      if (condition.has_p)
      {
        bc.addDof(p_dof.globalDof(node), condition.p);
        has_pressure_bc = true;
      }
    }
  }

  return bc;
}

DirichletCondition navierBoundary(const BlockFESpace& space)
{
  const Mesh& mesh = space.mesh();
  if (mesh.boundaryFacets().empty())
  {
    return cavityBoundary(space);
  }

  DirichletCondition bc;
  const auto         u_dof = space.field(0);
  const auto         p_dof = space.field(1);

  if (!bcs.empty())
  {
    bool               has_pressure_bc = false;
    DirichletCondition configured_bc =
        configuredBoundary(space, has_pressure_bc);
    if (!has_pressure_bc)
    {
      configured_bc.addDof(p_dof.globalDof(0), 0.0);
    }
    return configured_bc;
  }

  const auto inlet_nodes  = boundaryNodes(mesh, "inlet");
  const auto wall_nodes   = boundaryNodes(mesh, "wall");
  const auto outlet_nodes = boundaryNodes(mesh, "outlet");

  for (index_type node : wall_nodes)
  {
    bc.addDof(u_dof.globalDof(node, 0), 0.0);
    bc.addDof(u_dof.globalDof(node, 1), 0.0);
    if (u_dof.numComponents() > 2)
    {
      bc.addDof(u_dof.globalDof(node, 2), 0.0);
    }
  }

  real_type y_min = std::numeric_limits<real_type>::max();
  real_type y_max = std::numeric_limits<real_type>::lowest();
  for (index_type node : inlet_nodes)
  {
    const real_type y = mesh.node(node)[1];
    y_min             = std::min(y_min, y);
    y_max             = std::max(y_max, y);
  }

  const real_type height = y_max - y_min;
  for (index_type node : inlet_nodes)
  {
    real_type ux = inlet_velocity;
    if (height > 1.0e-14)
    {
      const real_type center = 0.5 * (y_min + y_max);
      const real_type eta    = 2.0 * (mesh.node(node)[1] - center) / height;
      ux                     = inlet_velocity * std::max<real_type>(0.0, 1.0 - eta * eta);
    }
    bc.addDof(u_dof.globalDof(node, 0), ux);
    bc.addDof(u_dof.globalDof(node, 1), 0.0);
    if (u_dof.numComponents() > 2)
    {
      bc.addDof(u_dof.globalDof(node, 2), 0.0);
    }
  }

  if (!outlet_nodes.empty())
  {
    for (index_type node : outlet_nodes)
    {
      bc.addDof(p_dof.globalDof(node), 0.0);
    }
  }
  else
  {
    bc.addDof(p_dof.globalDof(0), 0.0);
  }

  return bc;
}

void assembleSystem(const BlockFESpace& space,
                    const Vector&       x,
                    const Vector&       xp,
                    bool                initial,
                    SparseMatrix&       A,
                    Vector&             b,
                    real_type&          observed_max_cfl)
{
  const auto& element = space.field(0).space().finiteElement();
  const auto  quadrature =
      GaussQuadrature::make(element.referenceElement(), 2);
  ElementValues ev(element, quadrature);

  A.setZero();
  b.setZero();
  observed_max_cfl = 0.0;

  std::vector<QPState> qp_states(
      static_cast<std::size_t>(ev.numQuadraturePoints()));
  std::vector<index_type> elem_dofs;
  DenseMatrix Ke(space.numDofsPerElem(), space.numDofsPerElem());
  Vector      Fe(space.numDofsPerElem());

  for (index_type ic = 0; ic < space.mesh().numElems(); ++ic)
  {
    ev.reinit(space.mesh().cells()[static_cast<std::size_t>(ic)]);
    updateCellState(qp_states,
                    ev,
                    space,
                    ic,
                    x,
                    xp,
                    initial,
                    observed_max_cfl);

    Ke.setZero();
    Fe.setZero();

    assembleTransientLHS(ev, Ke);
    assembleAdvectionLHS(ev, qp_states, Ke);
    assembleViscousLHS(ev, Ke);
    assemblePressureVelocityCouplingLHS(ev, Ke);
    assembleStabilizationLHS(ev, qp_states, Ke);

    assembleTransientRHS(ev, qp_states, Fe);
    assembleAdvectionRHS(ev, qp_states, Fe);
    assembleViscousRHS(ev, qp_states, Fe);
    assembleStabilizationRHS(ev, qp_states, Fe);

    A.addLocalMatrix(ic, Ke);
    space.elemDofs(ic, elem_dofs);
    b.addLocalVector(elem_dofs, Fe);
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
                       Vector(mesh.numNodes()),
                       Vector(mesh.numNodes())};
  splitFields(x,
              space,
              mesh.numNodes(),
              snapshot.ux,
              snapshot.uy,
              snapshot.uz,
              snapshot.p);
  return snapshot;
}

} // namespace refem
