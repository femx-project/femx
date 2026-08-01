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

using BoundaryFacetSelector = std::function<bool(const Mesh::BoundaryFacet&)>;

/**
 * @brief Describe an axial boundary velocity profile.
 */
struct AxialVelocityProfile
{
  std::string type = "uniform";       ///< Profile type.
  Real        rad  = 0.0;             ///< Profile radius.
  Point3      cen  = {0.0, 0.0, 0.0}; ///< Profile center.
  Point3      nrm  = {1.0, 0.0, 0.0}; ///< Axial direction.
};

/**
 * @brief Compute the weighted center of selected boundary facets.
 *
 * @param[in] mesh  - Finite-element mesh.
 * @param[in] sel   - Boundary-facet selector.
 * @param[in] label - Boundary label used in diagnostics.
 * @return Weighted boundary center.
 * @throws std::runtime_error If validation fails.
 */
Point3 boundaryCenter(const Mesh&                  mesh,
                      const BoundaryFacetSelector& sel,
                      const std::string&           label = "boundary");

/**
 * @brief Compute the center of a physically tagged boundary.
 *
 * @param[in] mesh - Finite-element mesh.
 * @param[in] ptag - Physical boundary tag.
 * @return Weighted boundary center.
 * @throws std::runtime_error If validation fails.
 */
Point3 boundaryCenter(const Mesh& mesh, Index ptag);

/**
 * @brief Compute the center of a physically named boundary.
 *
 * @param[in] mesh  - Finite-element mesh.
 * @param[in] pname - Physical boundary name.
 * @return Weighted boundary center.
 * @throws std::runtime_error If validation fails.
 */
Point3 boundaryCenter(const Mesh& mesh, const std::string& pname);

/**
 * @brief Construct a uniform axial velocity profile.
 *
 * @param[in] nrm - Axial direction.
 * @return Uniform velocity profile.
 */
AxialVelocityProfile uniformProfile(const Point3& nrm);

/**
 * @brief Construct a Poiseuille axial velocity profile.
 *
 * @param[in] cen - Profile center.
 * @param[in] nrm - Axial direction.
 * @param[in] rad - Profile radius.
 * @return Poiseuille velocity profile.
 * @throws std::runtime_error If validation fails.
 */
AxialVelocityProfile poiseuilleProfile(const Point3& cen,
                                       const Point3& nrm,
                                       Real          rad);

/**
 * @brief Compute the scalar factor of an axial profile at a point.
 *
 * @param[in] prof - Velocity profile.
 * @param[in] p    - Evaluation point.
 * @return Profile factor.
 */
Real profileFactor(const AxialVelocityProfile& prof,
                   const Point3&               p);

/**
 * @brief Compute one velocity component at a point.
 *
 * @param[in] prof       - Velocity profile.
 * @param[in] p          - Evaluation point.
 * @param[in] peak_speed - Peak velocity magnitude.
 * @param[in] comp       - Component index.
 * @return Velocity component.
 */
Real velocityComponent(const AxialVelocityProfile& prof,
                       const Point3&               p,
                       Real                        peak_speed,
                       Index                       comp);

/**
 * @brief Convert a velocity quantity to peak speed.
 *
 * @param[in] qty          - Velocity quantity name.
 * @param[in] profile_type - Velocity profile type.
 * @param[in] val          - Input quantity.
 * @param[in] area         - Boundary area.
 * @param[in] mean_to_peak - Mean-to-peak conversion factor.
 * @return Peak speed.
 */
Real peakSpeed(const std::string& qty,
               const std::string& profile_type,
               Real               val,
               Real               area         = 1.0,
               Real               mean_to_peak = 2.0);

/**
 * @brief Compute a sinusoidal pulse factor.
 *
 * @param[in] time      - Evaluation time.
 * @param[in] amplitude - Pulse amplitude.
 * @param[in] per       - Pulse period.
 * @return Multiplicative pulse factor.
 */
Real sinePulseFactor(Real time, Real amplitude, Real per);

} // namespace fem
} // namespace femx
