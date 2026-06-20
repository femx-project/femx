#pragma once

#include <array>
#include <vector>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>

namespace femx
{
namespace inverse
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

} // namespace inverse
} // namespace femx
