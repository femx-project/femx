#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <femx/fem/VelocityProfile.hpp>

namespace femx
{
namespace fem
{

namespace
{

void requireValidComponent(Index comp)
{
  if (comp < 0 || comp >= 3)
  {
    throw std::runtime_error("Velocity profile component is out of range");
  }
}

void requirePositiveRadius(Real rad)
{
  if (rad <= 0.0)
  {
    throw std::runtime_error("Poiseuille velocity profile radius must be positive");
  }
}

Point3 facetCenter(const Mesh& mesh, const Mesh::BoundaryFacet& facet)
{
  Point3 cen = {0.0, 0.0, 0.0};
  for (Index in : facet.nids)
  {
    const Point3& point = mesh.node(in);
    for (Index id = 0; id < 3; ++id)
    {
      cen[id] += point[id];
    }
  }
  for (Index id = 0; id < 3; ++id)
  {
    cen[id] /= static_cast<Real>(facet.nids.size());
  }
  return cen;
}

Real facetWeight(const Mesh& mesh, const Mesh::BoundaryFacet& facet)
{
  if (facet.nids.size() == 2)
  {
    return distance(mesh.node(facet.nids[0]),
                    mesh.node(facet.nids[1]));
  }
  if (facet.nids.size() == 3)
  {
    return triArea(mesh.node(facet.nids[0]),
                   mesh.node(facet.nids[1]),
                   mesh.node(facet.nids[2]));
  }
  return 1.0;
}

} // namespace

Point3 boundaryCenter(const Mesh&                  mesh,
                      const BoundaryFacetSelector& sel,
                      const std::string&           label)
{
  Point3 cen          = {0.0, 0.0, 0.0};
  Real   total_weight = 0.0;

  for (const auto& facet : mesh.boundaryFacets())
  {
    if (!sel(facet) || facet.nids.empty())
    {
      continue;
    }

    const Point3 center_i = facetCenter(mesh, facet);
    const Real   wt       = facetWeight(mesh, facet);
    for (Index id = 0; id < 3; ++id)
    {
      cen[id] += wt * center_i[id];
    }
    total_weight += wt;
  }

  if (total_weight <= 0.0)
  {
    throw std::runtime_error("No boundary facets found for " + label);
  }

  for (Index id = 0; id < 3; ++id)
  {
    cen[id] /= total_weight;
  }
  return cen;
}

Point3 boundaryCenter(const Mesh& mesh, Index ptag)
{
  return boundaryCenter(
      mesh,
      [ptag](const Mesh::BoundaryFacet& facet)
      {
        return facet.ptag == ptag;
      },
      "physical tag " + std::to_string(ptag));
}

Point3 boundaryCenter(const Mesh& mesh, const std::string& pname)
{
  return boundaryCenter(
      mesh,
      [&pname](const Mesh::BoundaryFacet& facet)
      {
        return facet.pname == pname;
      },
      "physical name " + pname);
}

AxialVelocityProfile uniformProfile(const Point3& nrm)
{
  AxialVelocityProfile prof;
  prof.type = "uniform";
  prof.nrm  = unit(nrm);
  return prof;
}

AxialVelocityProfile poiseuilleProfile(const Point3& cen,
                                       const Point3& nrm,
                                       Real          rad)
{
  requirePositiveRadius(rad);

  AxialVelocityProfile prof;
  prof.type = "poiseuille";
  prof.cen  = cen;
  prof.nrm  = unit(nrm);
  prof.rad  = rad;
  return prof;
}

Real profileFactor(const AxialVelocityProfile& prof,
                   const Point3&               p)
{
  if (prof.type == "uniform")
  {
    return 1.0;
  }
  if (prof.type != "poiseuille")
  {
    throw std::runtime_error("Unsupported velocity profile type: "
                             + prof.type);
  }

  requirePositiveRadius(prof.rad);

  const Real radius2 = prof.rad * prof.rad;
  return std::max<Real>(
      0.0, 1.0 - radialSq(p, prof.cen, prof.nrm) / radius2);
}

Real velocityComponent(const AxialVelocityProfile& prof,
                       const Point3&               p,
                       Real                        peak_speed,
                       Index                       comp)
{
  requireValidComponent(comp);
  const Point3 nrm = unit(prof.nrm);
  return peak_speed * profileFactor(prof, p) * nrm[comp];
}

Real peakSpeed(const std::string& qty,
               const std::string& profile_type,
               Real               val,
               Real               area,
               Real               mean_to_peak)
{
  if (profile_type == "uniform")
  {
    if (qty == "flowrate")
    {
      if (area <= 0.0)
      {
        throw std::runtime_error("Flowrate velocity area must be positive");
      }
      return val / area;
    }
    if (qty == "mean_velocity" || qty == "bulk_speed"
        || qty == "max_velocity")
    {
      return val;
    }
  }
  else if (profile_type == "poiseuille")
  {
    if (qty == "flowrate")
    {
      if (area <= 0.0)
      {
        throw std::runtime_error("Flowrate velocity area must be positive");
      }
      return mean_to_peak * val / area;
    }
    if (qty == "mean_velocity" || qty == "bulk_speed")
    {
      return mean_to_peak * val;
    }
    if (qty == "max_velocity")
    {
      return val;
    }
  }
  else
  {
    throw std::runtime_error("Unsupported velocity profile type: "
                             + profile_type);
  }

  throw std::runtime_error("Unsupported velocity quantity: " + qty);
}

Real sinePulseFactor(Real time, Real amplitude, Real per)
{
  if (per <= 0.0)
  {
    throw std::runtime_error("Sine pulse period must be positive");
  }
  return 1.0 + amplitude * sin(2.0 * constants::PI * time / per);
}

} // namespace fem
} // namespace femx
