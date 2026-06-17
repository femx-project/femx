#include "BoundaryConditions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include <femx/fem/MixedFESpace.hpp>

namespace femx
{

Index lowerInterval(const Vector<Real>& points,
                    Real                x)
{
  const auto upper = std::upper_bound(points.begin(), points.end(), x);
  return static_cast<Index>(
      std::distance(points.begin(), upper) - 1);
}

Real periodicTime(const VelocityParams& velocity,
                  Real                  time)
{
  if (velocity.period <= 0.0)
  {
    return time;
  }

  const Real start = velocity.time.front();
  Real       local = std::fmod(time - start, velocity.period);
  if (local < 0.0)
  {
    local += velocity.period;
  }
  return start + local;
}

Real sampleVelocityConstant(const VelocityParams& velocity,
                            Real                  time)
{
  if (velocity.time.size() == 1 || time <= velocity.time.front())
  {
    return velocity.value.front();
  }
  if (time >= velocity.time.back())
  {
    return velocity.value.back();
  }

  return velocity.value[lowerInterval(velocity.time, time)];
}

Real sampleVelocityNearest(const VelocityParams& velocity,
                           Real                  time)
{
  if (velocity.time.size() == 1 || time <= velocity.time.front())
  {
    return velocity.value.front();
  }
  if (time >= velocity.time.back())
  {
    return velocity.value.back();
  }

  const Index i = lowerInterval(velocity.time, time);
  if (time - velocity.time[i] <= velocity.time[i + 1] - time)
  {
    return velocity.value[i];
  }
  return velocity.value[i + 1];
}

Real sampleVelocityLinear(const VelocityParams& velocity,
                          Real                  time)
{
  if (velocity.time.size() == 1 || time <= velocity.time.front())
  {
    return velocity.value.front();
  }
  if (time >= velocity.time.back())
  {
    return velocity.value.back();
  }

  const Index i  = lowerInterval(velocity.time, time);
  const Real  t0 = velocity.time[i];
  const Real  t1 = velocity.time[i + 1];
  const Real  a  = (time - t0) / (t1 - t0);
  return velocity.value[i]
         + a * (velocity.value[i + 1] - velocity.value[i]);
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

Real sampleVelocityCubic(const VelocityParams& velocity,
                         Real                  time)
{
  if (velocity.time.size() < 4)
  {
    return sampleVelocityLinear(velocity, time);
  }
  if (time <= velocity.time.front())
  {
    return velocity.value.front();
  }
  if (time >= velocity.time.back())
  {
    return velocity.value.back();
  }

  const Index i  = lowerInterval(velocity.time, time);
  Index       i0 = i;
  if (i > 0)
  {
    i0 = i - 1;
  }
  const Index i1 = i;
  const Index i2 = i + 1;
  Index       i3 = i + 2;
  if (i3 >= velocity.value.size())
  {
    i3 = velocity.period > 0.0 ? i3 - (velocity.value.size() - 1)
                               : velocity.value.size() - 1;
  }

  const Real a = (time - velocity.time[i1])
                 / (velocity.time[i2] - velocity.time[i1]);
  return catmullRom(velocity.value[i0],
                    velocity.value[i1],
                    velocity.value[i2],
                    velocity.value[i3],
                    a);
}

Real sampleVelocityValue(const VelocityParams& velocity,
                         Real                  time)
{
  const Real sample_time = periodicTime(velocity, time);
  if (velocity.interp == "constant")
  {
    return sampleVelocityConstant(velocity, sample_time);
  }
  if (velocity.interp == "nearest")
  {
    return sampleVelocityNearest(velocity, sample_time);
  }
  if (velocity.interp == "linear")
  {
    return sampleVelocityLinear(velocity, sample_time);
  }
  if (velocity.interp == "cubic")
  {
    return sampleVelocityCubic(velocity, sample_time);
  }
  throw std::runtime_error("Unsupported velocity interpolation: "
                           + velocity.interp);
}

Real norm3(Real x,
           Real y,
           Real z)
{
  return std::sqrt(x * x + y * y + z * z);
}

std::array<Real, 3> unitNormal(const VelocityParams& velocity)
{
  Real normal_mag2 = 0.0;
  for (Real comp : velocity.normal)
  {
    normal_mag2 += comp * comp;
  }
  const Real normal_mag = std::sqrt(normal_mag2);

  return {velocity.normal[0] / normal_mag,
          velocity.normal[1] / normal_mag,
          velocity.normal[2] / normal_mag};
}

std::array<Real, 3> boundaryCenter(const Mesh& mesh,
                                   Index       physical_tag)
{
  std::array<Real, 3> center       = {0.0, 0.0, 0.0};
  Real                total_weight = 0.0;

  for (const auto& facet : mesh.boundaryFacets())
  {
    if (facet.physical_tag != physical_tag || facet.node_ids.empty())
    {
      continue;
    }

    std::array<Real, 3> facet_center = {0.0, 0.0, 0.0};
    for (Index node_id : facet.node_ids)
    {
      const auto& point = mesh.node(node_id);
      for (Index d = 0; d < 3; ++d)
      {
        facet_center[static_cast<std::size_t>(d)] += point[d];
      }
    }
    for (Index d = 0; d < 3; ++d)
    {
      facet_center[static_cast<std::size_t>(d)] /=
          static_cast<Real>(facet.node_ids.size());
    }

    Real weight = 1.0;
    if (facet.node_ids.size() == 2)
    {
      const auto& x0 = mesh.node(facet.node_ids[0]);
      const auto& x1 = mesh.node(facet.node_ids[1]);
      weight         = norm3(x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]);
    }
    else if (facet.node_ids.size() == 3)
    {
      const auto& x0  = mesh.node(facet.node_ids[0]);
      const auto& x1  = mesh.node(facet.node_ids[1]);
      const auto& x2  = mesh.node(facet.node_ids[2]);
      const Real  e1x = x1[0] - x0[0];
      const Real  e1y = x1[1] - x0[1];
      const Real  e1z = x1[2] - x0[2];
      const Real  e2x = x2[0] - x0[0];
      const Real  e2y = x2[1] - x0[1];
      const Real  e2z = x2[2] - x0[2];
      const Real  nx  = e1y * e2z - e1z * e2y;
      const Real  ny  = e1z * e2x - e1x * e2z;
      const Real  nz  = e1x * e2y - e1y * e2x;
      weight          = 0.5 * norm3(nx, ny, nz);
    }

    for (Index d = 0; d < 3; ++d)
    {
      center[static_cast<std::size_t>(d)] +=
          weight * facet_center[static_cast<std::size_t>(d)];
    }
    total_weight += weight;
  }

  if (total_weight <= 0.0)
  {
    throw std::runtime_error(
        "No boundary facets found for physical tag "
        + std::to_string(physical_tag));
  }

  for (Index d = 0; d < 3; ++d)
  {
    center[static_cast<std::size_t>(d)] /= total_weight;
  }
  return center;
}

Real peakSpeed(const VelocityParams& velocity,
               Real                  scalar)
{
  if (velocity.profile.type == "uniform")
  {
    if (velocity.quantity == "flowrate")
    {
      return scalar / velocity.area;
    }
    return scalar;
  }

  if (velocity.quantity == "flowrate")
  {
    return 2.0 * scalar / velocity.area;
  }
  if (velocity.quantity == "mean_velocity")
  {
    return 2.0 * scalar;
  }
  return scalar;
}

struct VelocityEvalContext
{
  VelocityProfileParams profile;
  std::array<Real, 3>   normal     = {1.0, 0.0, 0.0};
  std::array<Real, 3>   center     = {0.0, 0.0, 0.0};
  Real                  peak_speed = 0.0;
};

VelocityEvalContext makeVelocityEvalContext(const VelocityParams& velocity,
                                            const Mesh&           mesh,
                                            Index                 physical_tag,
                                            Real                  time)
{
  VelocityEvalContext ctx;
  ctx.profile    = velocity.profile;
  ctx.normal     = unitNormal(velocity);
  ctx.peak_speed = peakSpeed(velocity, sampleVelocityValue(velocity, time));
  if (velocity.profile.type == "poiseuille")
  {
    ctx.center = velocity.profile.center
                     ? *velocity.profile.center
                     : boundaryCenter(mesh, physical_tag);
  }
  return ctx;
}

Real profileFactor(const VelocityEvalContext& ctx,
                   const Mesh::Node&          point)
{
  if (ctx.profile.type == "uniform")
  {
    return 1.0;
  }

  const Real dx = point[0] - ctx.center[0];
  const Real dy = point[1] - ctx.center[1];
  const Real dz = point[2] - ctx.center[2];
  const Real axial =
      dx * ctx.normal[0] + dy * ctx.normal[1] + dz * ctx.normal[2];
  Real radial2 = dx * dx + dy * dy + dz * dz - axial * axial;
  radial2      = std::max<Real>(0.0, radial2);

  const Real radius2 = ctx.profile.radius * ctx.profile.radius;
  return std::max<Real>(0.0, 1.0 - radial2 / radius2);
}

Real velocityComponent(const VelocityEvalContext& ctx,
                       const Mesh::Node&          point,
                       Index                      component)
{
  return ctx.peak_speed * profileFactor(ctx, point)
         * ctx.normal[static_cast<std::size_t>(component)];
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
    if (cond.velocity)
    {
      const auto ctx = makeVelocityEvalContext(*cond.velocity,
                                               u_dof.space().mesh(),
                                               cond.tag,
                                               time);
      if (u_dof.numComponents() < 3 && std::abs(ctx.normal[2]) > 1.0e-14)
      {
        throw std::runtime_error("3D velocity normal requires a 3D mesh");
      }

      for (Index d = 0; d < u_dof.numComponents(); ++d)
      {
        bc.addBoundary(u_dof, cond.tag, [ctx, d](const Mesh::Node& point, Real)
                       { return velocityComponent(ctx, point, d); },
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
