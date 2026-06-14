#include "BoundaryConditions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <femx/fem/MixedFESpace.hpp>

namespace femx
{

std::size_t lowerInterval(const std::vector<real_type>& points,
                          real_type                     x)
{
  const auto upper = std::upper_bound(points.begin(), points.end(), x);
  return static_cast<std::size_t>(
      std::distance(points.begin(), upper) - 1);
}

real_type sampleFlowRateLinear(const FlowRateParams& flow,
                               real_type             time)
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
  const real_type   t0 = flow.time[i];
  const real_type   t1 = flow.time[i + 1];
  const real_type   a  = (time - t0) / (t1 - t0);
  return flow.value[i] + a * (flow.value[i + 1] - flow.value[i]);
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

real_type sampleFlowRateCubic(const FlowRateParams& flow,
                              real_type             time)
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
  const std::size_t i3 =
      std::min<std::size_t>(i + 2, flow.value.size() - 1);
  const real_type a =
      (time - flow.time[i1]) / (flow.time[i2] - flow.time[i1]);
  return catmullRom(flow.value[i0],
                    flow.value[i1],
                    flow.value[i2],
                    flow.value[i3],
                    a);
}

real_type sampleFlowRate(const FlowRateParams& flow,
                         real_type             time)
{
  if (flow.interp == "linear")
  {
    return sampleFlowRateLinear(flow, time);
  }
  if (flow.interp == "cubic")
  {
    return sampleFlowRateCubic(flow, time);
  }
  throw std::runtime_error("Unsupported flowrate interpolation: "
                           + flow.interp);
}

std::array<real_type, 3> velFromFlow(const FlowRateParams& flow,
                                     real_type             time)
{
  const real_type q = sampleFlowRate(flow, time);

  real_type normal_mag2 = 0.0;
  for (real_type comp : flow.normal)
  {
    normal_mag2 += comp * comp;
  }
  const real_type normal_mag = std::sqrt(normal_mag2);
  const real_type speed      = q / flow.area;

  std::array<real_type, 3> vel{};
  for (std::size_t i = 0; i < vel.size(); ++i)
  {
    vel[i] = speed * flow.normal[i] / normal_mag;
  }
  return vel;
}

DirichletCondition makeBoundaryCondition(
    const MixedFESpace&           space,
    const std::vector<BCsParams>& bcs,
    real_type                     time)
{
  const auto u_dof = space.field(0);
  const auto p_dof = space.field(1);

  DirichletCondition bc;
  bool               has_pre_cond = false;

  for (const auto& cond : bcs)
  {
    std::array<real_type, 3> vel{};
    if (cond.flow)
    {
      vel = velFromFlow(*cond.flow, time);
      if (u_dof.numComponents() < 3
          && std::abs(vel[2]) > 1.0e-14)
      {
        throw std::runtime_error("3D flowrate normal requires a 3D mesh");
      }
    }

    if (cond.flow)
    {
      for (index_type d = 0; d < u_dof.numComponents(); ++d)
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
