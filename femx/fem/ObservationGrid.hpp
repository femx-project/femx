#pragma once

#include <array>
#include <vector>

#include <femx/core/Math.hpp>
#include <femx/core/Types.hpp>

namespace femx
{
namespace fem
{

/** @brief Background-grid points for sparse observation data. */
std::vector<Point3> observationGridPoints(
    const Point3&               lower,
    const Point3&               upper,
    const std::array<Index, 3>& counts);

/** @brief Background-grid points from origin and spacing. */
std::vector<Point3> observationGridPoints(
    const Point3&               origin,
    const std::array<Index, 3>& counts,
    const Point3&               spacing);

} // namespace fem
} // namespace femx
