#pragma once

#include <array>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

/** @brief Background-grid points for sparse observation data. */
Array<Point3> observationGridPoints(
    const Point3&               lower,
    const Point3&               upper,
    const std::array<Index, 3>& counts);

/** @brief Background-grid points from origin and spacing. */
Array<Point3> observationGridPoints(
    const Point3&               origin,
    const std::array<Index, 3>& counts,
    const Point3&               spacing);

} // namespace fem
} // namespace femx
