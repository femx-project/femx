#include "BoundaryConditions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <stdexcept>

#include <refem/fe/BlockFESpace.hpp>
#include <refem/mesh/Mesh.hpp>

namespace refem
{

std::set<index_type> boundaryNodes(const Mesh& mesh,
                                   index_type  tag)
{
  std::set<index_type> nodes;
  for (const auto& facet : mesh.boundaryFacets())
  {
    if (facet.physical_tag == tag)
    {
      for (index_type node : facet.node_ids)
      {
        nodes.insert(node);
      }
    }
  }
  return nodes;
}

std::size_t lowerInterval(const std::vector<real_type>& points,
                          real_type                     x)
{
  const auto upper = std::upper_bound(points.begin(), points.end(), x);
  return static_cast<std::size_t>(
      std::distance(points.begin(), upper) - 1);
}

real_type sampleFlowRateLinear(const FlowRateParams& flowrate,
                               real_type             time)
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

real_type sampleFlowRateConstant(const FlowRateParams& flowrate,
                                 real_type             time)
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

real_type sampleFlowRateNearest(const FlowRateParams& flowrate,
                                real_type             time)
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
  return 0.5
         * ((2.0 * y1) + (-y0 + y2) * a
            + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * a2
            + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * a3);
}

real_type sampleFlowRateCubic(const FlowRateParams& flowrate,
                              real_type             time)
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
  const std::size_t i3 =
      std::min<std::size_t>(i + 2, flowrate.value.size() - 1);
  const real_type a =
      (time - flowrate.time[i1]) / (flowrate.time[i2] - flowrate.time[i1]);
  return catmullRom(flowrate.value[i0],
                    flowrate.value[i1],
                    flowrate.value[i2],
                    flowrate.value[i3],
                    a);
}

real_type sampleFlowRate(const FlowRateParams& flowrate,
                         real_type             time)
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
  throw std::runtime_error("Unsupported flowrate interpolation: "
                           + flowrate.interpolate);
}

std::array<real_type, 3> flowRateVelocity(const FlowRateParams& flowrate,
                                          real_type             time)
{
  const real_type q = sampleFlowRate(flowrate, time);

  real_type normal_mag2 = 0.0;
  for (real_type component : flowrate.normal)
  {
    normal_mag2 += component * component;
  }
  const real_type normal_mag = std::sqrt(normal_mag2);
  const real_type speed      = q / flowrate.area;

  std::array<real_type, 3> velocity{};
  for (std::size_t i = 0; i < velocity.size(); ++i)
  {
    velocity[i] = speed * flowrate.normal[i] / normal_mag;
  }
  return velocity;
}

DirichletCondition makeBoundaryCondition(
    const BlockFESpace&           space,
    const std::vector<BCsParams>& bcs,
    real_type                     time)
{
  const Mesh& mesh  = space.mesh();
  const auto  u_dof = space.field(0);
  const auto  p_dof = space.field(1);

  DirichletCondition bc;
  bool               has_pressure_condition = false;

  for (const auto& condition : bcs)
  {
    const auto nodes = boundaryNodes(mesh, condition.tag);
    if (nodes.empty())
    {
      throw std::runtime_error(
          "No boundary facets found for physical tag " + std::to_string(condition.tag));
    }

    std::array<real_type, 3> flowrate_velocity{};
    if (condition.flowrate)
    {
      flowrate_velocity = flowRateVelocity(*condition.flowrate, time);
      if (u_dof.numComponents() < 3
          && std::abs(flowrate_velocity[2]) > 1.0e-14)
      {
        throw std::runtime_error("3D flowrate normal requires a 3D mesh");
      }
    }

    for (index_type node : nodes)
    {
      if (condition.flowrate)
      {
        for (index_type d = 0; d < u_dof.numComponents(); ++d)
        {
          bc.addDof(u_dof.globalDof(node, d),
                    flowrate_velocity[static_cast<std::size_t>(d)]);
        }
      }
      if (condition.ux)
      {
        bc.addDof(u_dof.globalDof(node, 0), *condition.ux);
      }
      if (condition.uy)
      {
        if (u_dof.numComponents() < 2)
        {
          throw std::runtime_error("uy boundary condition requires a 2D or 3D mesh");
        }
        bc.addDof(u_dof.globalDof(node, 1), *condition.uy);
      }
      if (condition.uz)
      {
        if (u_dof.numComponents() < 3)
        {
          throw std::runtime_error("uz boundary condition requires a 3D mesh");
        }
        bc.addDof(u_dof.globalDof(node, 2), *condition.uz);
      }
      if (condition.p)
      {
        bc.addDof(p_dof.globalDof(node), *condition.p);
        has_pressure_condition = true;
      }
    }
  }

  if (!has_pressure_condition)
  {
    bc.addDof(p_dof.globalDof(0), 0.0);
  }
  return bc;
}

} // namespace refem
