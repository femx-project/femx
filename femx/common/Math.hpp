#pragma once

#include <array>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>

namespace femx
{

/** @brief Three-dimensional point or vector. */
using Point3 = std::array<Real, 3>;

/**
 * @brief Compute the dot product of two Host vectors.
 *
 * @param[in] x - First vector.
 * @param[in] y - Second vector.
 * @return Dot product of `x` and `y`.
 * @throws std::runtime_error - If the vector sizes differ.
 */
Real dot(const HostVector<Real>& x, const HostVector<Real>& y);

/**
 * @brief Compute the dot product of two three-dimensional vectors.
 *
 * @param[in] x - First vector.
 * @param[in] y - Second vector.
 * @return Dot product of `x` and `y`.
 */
Real dot(const Point3& x, const Point3& y);

/**
 * @brief Compute the squared Euclidean norm of a Host vector.
 *
 * @param[in] x - Input vector.
 * @return Squared Euclidean norm of `x`.
 */
Real squaredNorm(const HostVector<Real>& x);

/**
 * @brief Compute the squared Euclidean norm of a three-dimensional vector.
 *
 * @param[in] x - Input vector.
 * @return Squared Euclidean norm of `x`.
 */
Real squaredNorm(const Point3& x);

/**
 * @brief Compute the Euclidean norm of a Host vector.
 *
 * @param[in] x - Input vector.
 * @return Euclidean norm of `x`.
 */
Real norm(const HostVector<Real>& x);

/**
 * @brief Compute the Euclidean norm of a three-dimensional vector.
 *
 * @param[in] x - Input vector.
 * @return Euclidean norm of `x`.
 */
Real norm(const Point3& x);

/**
 * @brief Compute the root mean square error between two Host vectors.
 *
 * @param[in] x - First vector.
 * @param[in] y - Second vector.
 * @return Root mean square error between `x` and `y`.
 * @throws std::runtime_error - If the vector sizes differ.
 */
Real rootMeanSquareError(const HostVector<Real>& x,
                         const HostVector<Real>& y);

/**
 * @brief Compute the component-wise difference of two Host vectors.
 *
 * @param[in] x - Vector to subtract from.
 * @param[in] y - Vector to subtract.
 * @return Component-wise difference `x - y`.
 * @throws std::runtime_error - If the vector sizes differ.
 */
HostVector<Real> difference(const HostVector<Real>& x,
                            const HostVector<Real>& y);

/**
 * @brief Compute the component-wise difference of two points.
 *
 * @param[in] x - Point to subtract from.
 * @param[in] y - Point to subtract.
 * @return Component-wise difference `x - y`.
 */
Point3 difference(const Point3& x, const Point3& y);

/**
 * @brief Compute the cross product of two three-dimensional vectors.
 *
 * @param[in] x - First vector.
 * @param[in] y - Second vector.
 * @return Cross product of `x` and `y`.
 */
Point3 cross(const Point3& x, const Point3& y);

/**
 * @brief Compute a unit vector in the direction of a vector.
 *
 * @param[in] x - Input vector.
 * @return Unit vector in the direction of `x`.
 * @throws std::runtime_error - If `x` is the zero vector.
 */
Point3 normalized(const Point3& x);

/**
 * @brief Compute the squared distance between two points.
 *
 * @param[in] x - First point.
 * @param[in] y - Second point.
 * @return Squared distance between `x` and `y`.
 */
Real squaredDistance(const Point3& x, const Point3& y);

/**
 * @brief Compute the distance between two points.
 *
 * @param[in] x - First point.
 * @param[in] y - Second point.
 * @return Distance between `x` and `y`.
 */
Real distance(const Point3& x, const Point3& y);

/**
 * @brief Compute the area of a triangle.
 *
 * @param[in] a - First vertex.
 * @param[in] b - Second vertex.
 * @param[in] c - Third vertex.
 * @return Area of the triangle.
 */
Real triangleArea(const Point3& a, const Point3& b, const Point3& c);

/**
 * @brief Compute squared distance from a point to a line.
 *
 * @param[in] point - Point whose distance to the line is computed.
 * @param[in] line_point - Point on the line.
 * @param[in] line_direction - Line direction.
 * @return Squared distance from `point` to the line.
 * @throws std::runtime_error - If `line_direction` is the zero vector.
 */
Real squaredDistanceToLine(const Point3& point,
                           const Point3& line_point,
                           const Point3& line_direction);

} // namespace femx
