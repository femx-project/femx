#pragma once

#include <array>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

using Point3 = std::array<Real, 3>;

Real dot(const HostVector& x, const HostVector& y);
Real dot(const Point3& x, const Point3& y);

Real squaredNorm(const HostVector& x);
Real squaredNorm(const Point3& x);

Real norm(const HostVector& x);
Real norm(const Point3& x);

Real rmse(const HostVector& x, const HostVector& y);

/** @brief Component-wise difference x - y. */
HostVector difference(const HostVector& x, const HostVector& y);

/** @brief Component-wise difference x - y. */
Point3 difference(const Point3& x, const Point3& y);

Point3 cross(const Point3& x, const Point3& y);

/** @brief Unit vector in the direction of x. */
Point3 unit(const Point3& x);

Real sqDist(const Point3& x, const Point3& y);
Real distance(const Point3& x, const Point3& y);
Real triArea(const Point3& a, const Point3& b, const Point3& c);

/** @brief Squared distance from point to the line through origin along axis. */
Real radialSq(const Point3& point, const Point3& origin, const Point3& axis);

} // namespace femx
