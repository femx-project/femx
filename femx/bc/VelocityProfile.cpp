#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <femx/bc/VelocityProfile.hpp>

namespace femx::bc
{

namespace
{

constexpr Real kPi = 3.141592653589793238462643383279502884;

void requireValidComponent(Index component)
{
  if (component < 0 || component >= 3)
  {
    throw std::runtime_error("Velocity profile component is out of range");
  }
}

void requirePositiveRadius(Real radius)
{
  if (radius <= 0.0)
  {
    throw std::runtime_error("Poiseuille velocity profile radius must be positive");
  }
}

Point3 facetCenter(const Mesh& mesh, const Mesh::BoundaryFacet& facet)
{
  Point3 center = {0.0, 0.0, 0.0};
  for (Index node_id : facet.node_ids)
  {
    const Point3& point = mesh.node(node_id);
    for (Index d = 0; d < 3; ++d)
    {
      center[static_cast<std::size_t>(d)] +=
          point[static_cast<std::size_t>(d)];
    }
  }
  for (Index d = 0; d < 3; ++d)
  {
    center[static_cast<std::size_t>(d)] /=
        static_cast<Real>(facet.node_ids.size());
  }
  return center;
}

Real facetWeight(const Mesh& mesh, const Mesh::BoundaryFacet& facet)
{
  if (facet.node_ids.size() == 2)
  {
    return distance(mesh.node(facet.node_ids[0]),
                    mesh.node(facet.node_ids[1]));
  }
  if (facet.node_ids.size() == 3)
  {
    return triArea(mesh.node(facet.node_ids[0]),
                   mesh.node(facet.node_ids[1]),
                   mesh.node(facet.node_ids[2]));
  }
  return 1.0;
}

} // namespace

Point3 boundaryCenter(const Mesh&                  mesh,
                      const BoundaryFacetSelector& selector,
                      const std::string&           label)
{
  Point3 center       = {0.0, 0.0, 0.0};
  Real   total_weight = 0.0;

  for (const auto& facet : mesh.boundaryFacets())
  {
    if (!selector(facet) || facet.node_ids.empty())
    {
      continue;
    }

    const Point3 center_i = facetCenter(mesh, facet);
    const Real   weight   = facetWeight(mesh, facet);
    for (Index d = 0; d < 3; ++d)
    {
      center[static_cast<std::size_t>(d)] +=
          weight * center_i[static_cast<std::size_t>(d)];
    }
    total_weight += weight;
  }

  if (total_weight <= 0.0)
  {
    throw std::runtime_error("No boundary facets found for " + label);
  }

  for (Index d = 0; d < 3; ++d)
  {
    center[static_cast<std::size_t>(d)] /= total_weight;
  }
  return center;
}

Point3 boundaryCenter(const Mesh& mesh, Index physical_tag)
{
  return boundaryCenter(
      mesh,
      [physical_tag](const Mesh::BoundaryFacet& facet)
      {
        return facet.physical_tag == physical_tag;
      },
      "physical tag " + std::to_string(physical_tag));
}

Point3 boundaryCenter(const Mesh& mesh, const std::string& physical_name)
{
  return boundaryCenter(
      mesh,
      [&physical_name](const Mesh::BoundaryFacet& facet)
      {
        return facet.physical_name == physical_name;
      },
      "physical name " + physical_name);
}

AxialVelocityProfile uniformProfile(const Point3& normal)
{
  AxialVelocityProfile profile;
  profile.type   = "uniform";
  profile.normal = unit(normal);
  return profile;
}

AxialVelocityProfile poiseuilleProfile(const Point3& center,
                                       const Point3& normal,
                                       Real          radius)
{
  requirePositiveRadius(radius);

  AxialVelocityProfile profile;
  profile.type   = "poiseuille";
  profile.center = center;
  profile.normal = unit(normal);
  profile.radius = radius;
  return profile;
}

Real profileFactor(const AxialVelocityProfile& profile,
                   const Point3&               point)
{
  if (profile.type == "uniform")
  {
    return 1.0;
  }
  if (profile.type != "poiseuille")
  {
    throw std::runtime_error("Unsupported velocity profile type: "
                             + profile.type);
  }

  requirePositiveRadius(profile.radius);

  const Real radius2 = profile.radius * profile.radius;
  return std::max<Real>(
      0.0, 1.0 - radialSq(point, profile.center, profile.normal) / radius2);
}

Real velocityComponent(const AxialVelocityProfile& profile,
                       const Point3&               point,
                       Real                        peak_speed,
                       Index                       component)
{
  requireValidComponent(component);
  const Point3 normal = unit(profile.normal);
  return peak_speed * profileFactor(profile, point)
         * normal[static_cast<std::size_t>(component)];
}

Real peakSpeed(const std::string& quantity,
               const std::string& profile_type,
               Real               value,
               Real               area,
               Real               mean_to_peak)
{
  if (profile_type == "uniform")
  {
    if (quantity == "flowrate")
    {
      if (area <= 0.0)
      {
        throw std::runtime_error("Flowrate velocity area must be positive");
      }
      return value / area;
    }
    if (quantity == "mean_velocity" || quantity == "bulk_speed"
        || quantity == "max_velocity")
    {
      return value;
    }
  }
  else if (profile_type == "poiseuille")
  {
    if (quantity == "flowrate")
    {
      if (area <= 0.0)
      {
        throw std::runtime_error("Flowrate velocity area must be positive");
      }
      return mean_to_peak * value / area;
    }
    if (quantity == "mean_velocity" || quantity == "bulk_speed")
    {
      return mean_to_peak * value;
    }
    if (quantity == "max_velocity")
    {
      return value;
    }
  }
  else
  {
    throw std::runtime_error("Unsupported velocity profile type: "
                             + profile_type);
  }

  throw std::runtime_error("Unsupported velocity quantity: " + quantity);
}

Real sinePulseFactor(Real time, Real amplitude, Real period)
{
  if (period <= 0.0)
  {
    throw std::runtime_error("Sine pulse period must be positive");
  }
  return 1.0 + amplitude * std::sin(2.0 * kPi * time / period);
}

} // namespace femx::bc
