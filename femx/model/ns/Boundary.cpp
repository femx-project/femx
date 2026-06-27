#include "Boundary.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/VelocityProfile.hpp>

using namespace std;
using namespace femx;
using namespace femx::fem;

namespace femx::model::ns
{

Index lowerInterval(const Vector<Real>& pts,
                    Real                x)
{
  using std::distance;
  const auto upper = upper_bound(pts.begin(), pts.end(), x);
  return static_cast<Index>(
      distance(pts.begin(), upper) - 1);
}

Real periodicTime(const VelocityParams& velocity,
                  Real                  time)
{
  if (velocity.per <= 0.0)
  {
    return time;
  }

  const Real start = velocity.time.front();
  Real       local = fmod(time - start, velocity.per);
  if (local < 0.0)
  {
    local += velocity.per;
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
    i3 = velocity.per > 0.0 ? i3 - (velocity.value.size() - 1)
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
  throw runtime_error("Unsupported velocity interpolation: "
                      + velocity.interp);
}

struct VelocityEvalContext
{
  AxialVelocityProfile prof;
  Real                 peak_speed = 0.0;
};

VelocityEvalContext makeVelocityEvalContext(const VelocityParams& velocity,
                                            const Mesh&           mesh,
                                            Index                 ptag,
                                            Real                  time)
{
  VelocityEvalContext ctx;
  ctx.peak_speed = peakSpeed(velocity.qty,
                             velocity.prof.type,
                             sampleVelocityValue(velocity, time),
                             velocity.area,
                             2.0);
  if (velocity.prof.type == "poiseuille")
  {
    ctx.prof = poiseuilleProfile(
        velocity.prof.cen ? *velocity.prof.cen
                          : boundaryCenter(mesh, ptag),
        velocity.nrm,
        velocity.prof.rad);
  }
  else
  {
    ctx.prof = uniformProfile(velocity.nrm);
  }
  return ctx;
}

Real velocityComponent(const VelocityEvalContext& ctx,
                       const Mesh::Node&          point,
                       Index                      comp)
{
  return velocityComponent(
      ctx.prof, point, ctx.peak_speed, comp);
}

DirichletCondition makeBoundaryCondition(
    const MixedFESpace&      space,
    const Vector<BCsParams>& bcs,
    Real                     time)
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
      if (u_dof.numComponents() < 3
          && abs(ctx.prof.nrm[2]) > 1.0e-14)
      {
        throw runtime_error("3D velocity normal requires a 3D mesh");
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
        throw runtime_error("uy boundary condition requires a 2D or 3D mesh");
      }
      bc.addBoundary(u_dof, cond.tag, *cond.uy, time, 1);
    }
    if (cond.uz)
    {
      if (u_dof.numComponents() < 3)
      {
        throw runtime_error("uz boundary condition requires a 3D mesh");
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

} // namespace femx::model::ns
