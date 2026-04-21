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
#include <refem/linalg/LocalAssembler.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/mesh/Mesh.hpp>

namespace refem
{

bool near(real_type a, real_type b)
{
  return std::abs(a - b) <= 1.0e-12;
}

real_type timeFactor(const BoundaryConditionSpec::TimeProfile& profile,
                     real_type                                 time)
{
  if (profile.profile == "constant")
  {
    return profile.value;
  }
  if (profile.profile == "ramp")
  {
    if (near(profile.t0, profile.t1))
    {
      return time < profile.t0 ? profile.from : profile.to;
    }
    const real_type alpha =
        std::max<real_type>(0.0,
                            std::min<real_type>(1.0, (time - profile.t0) / (profile.t1 - profile.t0)));
    return profile.from + alpha * (profile.to - profile.from);
  }
  if (profile.profile == "sin")
  {
    constexpr real_type pi = 3.141592653589793238462643383279502884;
    return profile.mean + profile.amplitude * std::sin(2.0 * pi * profile.frequency * time + profile.phase);
  }
  throw std::runtime_error("Unsupported boundary time profile: " + profile.profile);
}

std::size_t lowerInterval(const std::vector<real_type>& points,
                          real_type                     x)
{
  const auto upper = std::upper_bound(points.begin(), points.end(), x);
  return static_cast<std::size_t>(
      std::distance(points.begin(), upper) - 1);
}

real_type sampleFlowRateLinear(const BoundaryConditionSpec::FlowRate& flowrate,
                               real_type                              time)
{
  if (flowrate.time.size() == 1 || time <= flowrate.time.front())
  {
    return flowrate.value.front();
  }
  if (time >= flowrate.time.back())
  {
    return flowrate.value.back();
  }

  const std::size_t i  = lowerInterval(flowrate.time, time);
  const real_type   t0 = flowrate.time[i];
  const real_type   t1 = flowrate.time[i + 1];
  const real_type   a  = (time - t0) / (t1 - t0);
  return flowrate.value[i] + a * (flowrate.value[i + 1] - flowrate.value[i]);
}

real_type sampleFlowRateConstant(const BoundaryConditionSpec::FlowRate& flowrate,
                                 real_type                              time)
{
  if (flowrate.time.size() == 1 || time <= flowrate.time.front())
  {
    return flowrate.value.front();
  }
  if (time >= flowrate.time.back())
  {
    return flowrate.value.back();
  }
  return flowrate.value[lowerInterval(flowrate.time, time)];
}

real_type sampleFlowRateNearest(const BoundaryConditionSpec::FlowRate& flowrate,
                                real_type                              time)
{
  if (flowrate.time.size() == 1 || time <= flowrate.time.front())
  {
    return flowrate.value.front();
  }
  if (time >= flowrate.time.back())
  {
    return flowrate.value.back();
  }

  const std::size_t i  = lowerInterval(flowrate.time, time);
  const real_type   dl = time - flowrate.time[i];
  const real_type   dr = flowrate.time[i + 1] - time;
  return dl <= dr ? flowrate.value[i] : flowrate.value[i + 1];
}

real_type catmullRom(real_type y0,
                     real_type y1,
                     real_type y2,
                     real_type y3,
                     real_type a)
{
  const real_type a2 = a * a;
  const real_type a3 = a2 * a;
  return 0.5 * ((2.0 * y1) + (-y0 + y2) * a + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * a2 + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * a3);
}

real_type sampleFlowRateCubic(const BoundaryConditionSpec::FlowRate& flowrate,
                              real_type                              time)
{
  if (flowrate.time.size() < 4)
  {
    return sampleFlowRateLinear(flowrate, time);
  }
  if (time <= flowrate.time.front())
  {
    return flowrate.value.front();
  }
  if (time >= flowrate.time.back())
  {
    return flowrate.value.back();
  }

  const std::size_t i  = lowerInterval(flowrate.time, time);
  const std::size_t i0 = i == 0 ? i : i - 1;
  const std::size_t i1 = i;
  const std::size_t i2 = i + 1;
  const std::size_t i3 = std::min<std::size_t>(i + 2, flowrate.value.size() - 1);
  const real_type   a =
      (time - flowrate.time[i1]) / (flowrate.time[i2] - flowrate.time[i1]);
  return catmullRom(flowrate.value[i0],
                    flowrate.value[i1],
                    flowrate.value[i2],
                    flowrate.value[i3],
                    a);
}

real_type sampleFlowRate(const BoundaryConditionSpec::FlowRate& flowrate,
                         real_type                              time)
{
  if (flowrate.interpolate == "constant")
  {
    return sampleFlowRateConstant(flowrate, time);
  }
  if (flowrate.interpolate == "nearest")
  {
    return sampleFlowRateNearest(flowrate, time);
  }
  if (flowrate.interpolate == "linear")
  {
    return sampleFlowRateLinear(flowrate, time);
  }
  if (flowrate.interpolate == "cubic")
  {
    return sampleFlowRateCubic(flowrate, time);
  }
  throw std::runtime_error("Unsupported flowrate interpolation: " + flowrate.interpolate);
}

std::array<real_type, dim> flowRateVelocity(
    const BoundaryConditionSpec::FlowRate& flowrate,
    real_type                              time)
{
  const real_type q = sampleFlowRate(flowrate, time);

  real_type normal_mag2 = 0.0;
  for (real_type component : flowrate.normal)
  {
    normal_mag2 += component * component;
  }
  const real_type normal_mag = std::sqrt(normal_mag2);
  const real_type speed      = q / flowrate.area;

  std::array<real_type, dim> velocity{};
  for (index_type d = 0; d < dim; ++d)
  {
    const std::size_t i = static_cast<std::size_t>(d);
    velocity[i]         = speed * flowrate.normal[i] / normal_mag;
  }
  return velocity;
}

std::array<real_type, 2> boundaryCoordinateRange(const Mesh&                 mesh,
                                                 const std::set<index_type>& nodes,
                                                 index_type                  axis)
{
  if (axis < 0 || axis >= mesh.dim())
  {
    throw std::runtime_error("Boundary profile axis exceeds mesh dimension");
  }

  std::array<real_type, 2> range{
      std::numeric_limits<real_type>::max(),
      std::numeric_limits<real_type>::lowest()};
  for (index_type node : nodes)
  {
    const real_type coordinate = mesh.node(node)[axis];
    range[0]                   = std::min(range[0], coordinate);
    range[1]                   = std::max(range[1], coordinate);
  }
  return range;
}

std::array<real_type, 2> boundaryValueRange(
    const Mesh&                         mesh,
    const std::set<index_type>&         nodes,
    const BoundaryConditionSpec::Value& value)
{
  if (value.profile == "parabolic")
  {
    return boundaryCoordinateRange(mesh, nodes, value.axis);
  }
  return {0.0, 0.0};
}

real_type boundaryValue(const BoundaryConditionSpec::Value& value,
                        const Mesh::Node&                   point,
                        const std::array<real_type, 2>&     range,
                        real_type                           time)
{
  real_type spatial = value.value;
  if (value.profile == "parabolic")
  {
    const real_type height = range[1] - range[0];
    if (height > 1.0e-14)
    {
      const real_type center = 0.5 * (range[0] + range[1]);
      const real_type eta    = 2.0 * (point[value.axis] - center) / height;
      spatial                = value.value * std::max<real_type>(0.0, 1.0 - eta * eta);
    }
  }
  else if (value.profile != "constant")
  {
    throw std::runtime_error("Unsupported boundary value profile: " + value.profile);
  }
  return spatial * timeFactor(value.time, time);
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
                                      real_type           time,
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

    std::array<real_type, 2> ux_range{0.0, 0.0};
    std::array<real_type, 2> uy_range{0.0, 0.0};
    std::array<real_type, 2> uz_range{0.0, 0.0};
    std::array<real_type, 2> p_range{0.0, 0.0};

    if (condition.has_ux)
    {
      ux_range = boundaryValueRange(mesh, nodes, condition.ux);
    }
    if (condition.has_uy)
    {
      uy_range = boundaryValueRange(mesh, nodes, condition.uy);
    }
    if (condition.has_uz)
    {
      uz_range = boundaryValueRange(mesh, nodes, condition.uz);
    }
    if (condition.has_p)
    {
      p_range = boundaryValueRange(mesh, nodes, condition.p);
    }

    std::array<real_type, dim> flowrate_velocity{};
    if (condition.has_flowrate)
    {
      flowrate_velocity = flowRateVelocity(condition.flowrate, time);
      if (u_dof.numComponents() < 3 && std::abs(flowrate_velocity[2]) > 1.0e-14)
      {
        throw std::runtime_error("3D flowrate normal requires a 3D mesh");
      }
    }

    for (index_type node : nodes)
    {
      if (condition.has_flowrate)
      {
        for (index_type d = 0; d < u_dof.numComponents(); ++d)
        {
          bc.addDof(u_dof.globalDof(node, d),
                    flowrate_velocity[static_cast<std::size_t>(d)]);
        }
      }
      if (condition.has_ux)
      {
        bc.addDof(u_dof.globalDof(node, 0),
                  boundaryValue(condition.ux,
                                mesh.node(node),
                                ux_range,
                                time));
      }
      if (condition.has_uy)
      {
        bc.addDof(u_dof.globalDof(node, 1),
                  boundaryValue(condition.uy,
                                mesh.node(node),
                                uy_range,
                                time));
      }
      if (condition.has_uz)
      {
        if (u_dof.numComponents() < 3)
        {
          throw std::runtime_error("uz boundary condition requires a 3D mesh");
        }
        bc.addDof(u_dof.globalDof(node, 2),
                  boundaryValue(condition.uz,
                                mesh.node(node),
                                uz_range,
                                time));
      }
      if (condition.has_p)
      {
        bc.addDof(p_dof.globalDof(node),
                  boundaryValue(condition.p,
                                mesh.node(node),
                                p_range,
                                time));
        has_pressure_bc = true;
      }
    }
  }

  return bc;
}

DirichletCondition getBoundary(const BlockFESpace& space,
                               real_type           time)
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
        configuredBoundary(space, time, has_pressure_bc);
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
  const index_type nq = quadrature.size();

  A.setZero();
  b.setZero();
  observed_max_cfl = 0.0;

#pragma omp parallel reduction(max : observed_max_cfl)
  {
    ElementValues        ev(element, quadrature);
    std::vector<QPState> qp_states(static_cast<std::size_t>(nq));
    LocalAssembler       assembler(space,
                             A.pattern(),
                             LocalAssembler::AssemblyPolicy::Atomic);
    DenseMatrix          Ke(space.numDofsPerElem(), space.numDofsPerElem());
    Vector               Fe(space.numDofsPerElem());

#pragma omp for
    for (index_type ic = 0; ic < space.mesh().numElems(); ++ic)
    {
      ev.reinit(space.mesh().cell(ic));
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

      assembler.addLocalMatrix(ic, Ke, A);
      assembler.addLocalVector(ic, Fe, b);
    }
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
