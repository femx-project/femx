#include "BoundaryConditions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <femx/fem/MixedFESpace.hpp>

namespace femx
{

std::size_t lowerInterval(const std::vector<Real>& points,
                          Real                     x)
{
  const auto upper = std::upper_bound(points.begin(), points.end(), x);
  return static_cast<std::size_t>(
      std::distance(points.begin(), upper) - 1);
}

Real sampleFlowRateLinear(const FlowRateParams& flow,
                          Real                  time)
{
  if (flow.time.size() == 1 || time <= flow.time.front())
  {
    return flow.value.front();
  }
  if (time >= flow.time.back())
  {
    return flow.value.back();
  }

  const std::size_t i  = lowerInterval(flow.time, time);
  const Real        t0 = flow.time[i];
  const Real        t1 = flow.time[i + 1];
  const Real        a  = (time - t0) / (t1 - t0);
  return flow.value[i] + a * (flow.value[i + 1] - flow.value[i]);
}

Real catmullRom(Real y0,
                Real y1,
                Real y2,
                Real y3,
                Real a)
{
  const Real a2 = a * a;
  const Real a3 = a2 * a;
  return 0.5
         * ((2.0 * y1) + (-y0 + y2) * a
            + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * a2
            + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * a3);
}

Real sampleFlowRateCubic(const FlowRateParams& flow,
                         Real                  time)
{
  if (flow.time.size() < 4)
  {
    return sampleFlowRateLinear(flow, time);
  }
  if (time <= flow.time.front())
  {
    return flow.value.front();
  }
  if (time >= flow.time.back())
  {
    return flow.value.back();
  }

  const std::size_t i  = lowerInterval(flow.time, time);
  std::size_t       i0 = i;
  if (i > 0)
  {
    i0 = i - 1;
  }
  const std::size_t i1 = i;
  const std::size_t i2 = i + 1;
  const std::size_t i3 = std::min<std::size_t>(i + 2, flow.value.size() - 1);

  const Real a = (time - flow.time[i1]) / (flow.time[i2] - flow.time[i1]);
  return catmullRom(flow.value[i0],
                    flow.value[i1],
                    flow.value[i2],
                    flow.value[i3],
                    a);
}

Real sampleFlowRate(const FlowRateParams& flow,
                    Real                  time)
{
  if (flow.interp == "linear")
  {
    return sampleFlowRateLinear(flow, time);
  }
  if (flow.interp == "cubic")
  {
    return sampleFlowRateCubic(flow, time);
  }
  throw std::runtime_error("Unsupported flowrate interpolation: " + flow.interp);
}

std::array<Real, 3> velFromFlow(const FlowRateParams& flow,
                                Real                  time)
{
  const Real q = sampleFlowRate(flow, time);

  Real normal_mag2 = 0.0;
  for (Real comp : flow.normal)
  {
    normal_mag2 += comp * comp;
  }
  const Real normal_mag = std::sqrt(normal_mag2);
  const Real speed      = q / flow.area;

  std::array<Real, 3> vel{};
  for (std::size_t i = 0; i < vel.size(); ++i)
  {
    vel[i] = speed * flow.normal[i] / normal_mag;
  }
  return vel;
}

DirichletCondition makeBoundaryCondition(
    const MixedFESpace&           space,
    const std::vector<BCsParams>& bcs,
    Real                          time)
{
  const auto u_dof = space.field(0);
  const auto p_dof = space.field(1);

  DirichletCondition bc;
  bool               has_pre_cond = false;

  for (const auto& cond : bcs)
  {
    std::array<Real, 3> vel{};
    if (cond.flow)
    {
      vel = velFromFlow(*cond.flow, time);
      if (u_dof.numComponents() < 3 && std::abs(vel[2]) > 1.0e-14)
      {
        throw std::runtime_error("3D flowrate normal requires a 3D mesh");
      }
    }

    if (cond.flow)
    {
      for (Index d = 0; d < u_dof.numComponents(); ++d)
      {
        bc.addBoundary(u_dof,
                       cond.tag,
                       vel[static_cast<std::size_t>(d)],
                       time,
                       d);
      }
    }
    if (cond.ux)
    {
      bc.addBoundary(u_dof, cond.tag, *cond.ux, time, 0);
    }
    if (cond.uy)
    {
      if (u_dof.numComponents() < 2)
      {
        throw std::runtime_error("uy boundary condition requires a 2D or 3D mesh");
      }
      bc.addBoundary(u_dof, cond.tag, *cond.uy, time, 1);
    }
    if (cond.uz)
    {
      if (u_dof.numComponents() < 3)
      {
        throw std::runtime_error("uz boundary condition requires a 3D mesh");
      }
      bc.addBoundary(u_dof, cond.tag, *cond.uz, time, 2);
    }
    if (cond.p)
    {
      bc.addBoundary(p_dof, cond.tag, *cond.p, time);
      has_pre_cond = true;
    }
  }

  if (!has_pre_cond)
  {
    bc.addDof(p_dof.globalDof(0), 0.0);
  }
  return bc;
}

} // namespace femx
