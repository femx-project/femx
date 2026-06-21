#pragma once

#include <functional>
#include <string>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/Mesh.hpp>

namespace femx::fem
{

using BoundaryFacetSelector =
    std::function<bool(const Mesh::BoundaryFacet&)>;

struct AxialVelocityProfile
{
  std::string type   = "uniform";
  Real        radius = 0.0;
  Point3      center = {0.0, 0.0, 0.0};
  Point3      normal = {1.0, 0.0, 0.0};
};

Point3 boundaryCenter(const Mesh&                  mesh,
                      const BoundaryFacetSelector& selector,
                      const std::string&           label = "boundary");

Point3 boundaryCenter(const Mesh& mesh, Index physical_tag);

Point3 boundaryCenter(const Mesh& mesh, const std::string& physical_name);

AxialVelocityProfile uniformProfile(const Point3& normal);

AxialVelocityProfile poiseuilleProfile(const Point3& center,
                                       const Point3& normal,
                                       Real          radius);

Real profileFactor(const AxialVelocityProfile& profile,
                   const Point3&               point);

Real velocityComponent(const AxialVelocityProfile& profile,
                       const Point3&               point,
                       Real                        peak_speed,
                       Index                       component);

Real peakSpeed(const std::string& quantity,
               const std::string& profile_type,
               Real               value,
               Real               area         = 1.0,
               Real               mean_to_peak = 2.0);

Real sinePulseFactor(Real time, Real amplitude, Real period);

} // namespace femx::fem
