#pragma once

#include <functional>
#include <string>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/Mesh.hpp>

namespace femx
{
namespace fem
{

using BoundaryFacetSelector =
    std::function<bool(const Mesh::BoundaryFacet&)>;

struct AxialVelocityProfile
{
  std::string type = "uniform";       ///< Profile type.
  Real        rad  = 0.0;             ///< Profile radius.
  Point3      cen  = {0.0, 0.0, 0.0}; ///< Profile center.
  Point3      nrm  = {1.0, 0.0, 0.0}; ///< Axial direction.
};

Point3 boundaryCenter(const Mesh&                  mesh,
                      const BoundaryFacetSelector& sel,
                      const std::string&           label = "boundary");

Point3 boundaryCenter(const Mesh& mesh, Index ptag);

Point3 boundaryCenter(const Mesh& mesh, const std::string& pname);

AxialVelocityProfile uniformProfile(const Point3& nrm);

AxialVelocityProfile poiseuilleProfile(const Point3& cen,
                                       const Point3& nrm,
                                       Real          rad);

Real profileFactor(const AxialVelocityProfile& prof,
                   const Point3&               point);

Real velocityComponent(const AxialVelocityProfile& prof,
                       const Point3&               point,
                       Real                        peak_speed,
                       Index                       comp);

Real peakSpeed(const std::string& qty,
               const std::string& profile_type,
               Real               value,
               Real               area         = 1.0,
               Real               mean_to_peak = 2.0);

Real sinePulseFactor(Real time, Real amplitude, Real per);

} // namespace fem
} // namespace femx
