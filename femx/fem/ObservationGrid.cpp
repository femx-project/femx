#include <stdexcept>

#include <femx/fem/ObservationGrid.hpp>

namespace femx
{
namespace fem
{

namespace
{

void checkCounts(const std::array<Index, 3>& counts)
{
  for (Index count : counts)
  {
    if (count <= 0)
    {
      throw std::runtime_error(
          "observationGridPoints received non-positive grid count");
    }
  }
}

Real coordFromBounds(Real  lower,
                     Real  upper,
                     Index count,
                     Index i)
{
  if (count == 1)
  {
    return 0.5 * (lower + upper);
  }
  return lower + (upper - lower) * static_cast<Real>(i) / static_cast<Real>(count - 1);
}

} // namespace

std::vector<Point3> observationGridPoints(
    const Point3&               lower,
    const Point3&               upper,
    const std::array<Index, 3>& counts)
{
  checkCounts(counts);

  std::vector<Point3> points;
  points.reserve(static_cast<std::size_t>(counts[0] * counts[1] * counts[2]));

  for (Index k = 0; k < counts[2]; ++k)
  {
    for (Index j = 0; j < counts[1]; ++j)
    {
      for (Index i = 0; i < counts[0]; ++i)
      {
        points.push_back(
            {coordFromBounds(lower[0], upper[0], counts[0], i),
             coordFromBounds(lower[1], upper[1], counts[1], j),
             coordFromBounds(lower[2], upper[2], counts[2], k)});
      }
    }
  }
  return points;
}

std::vector<Point3> observationGridPoints(
    const Point3&               origin,
    const std::array<Index, 3>& counts,
    const Point3&               spacing)
{
  checkCounts(counts);

  std::vector<Point3> points;
  points.reserve(static_cast<std::size_t>(counts[0] * counts[1] * counts[2]));

  for (Index k = 0; k < counts[2]; ++k)
  {
    for (Index j = 0; j < counts[1]; ++j)
    {
      for (Index i = 0; i < counts[0]; ++i)
      {
        points.push_back({origin[0] + spacing[0] * static_cast<Real>(i),
                          origin[1] + spacing[1] * static_cast<Real>(j),
                          origin[2] + spacing[2] * static_cast<Real>(k)});
      }
    }
  }
  return points;
}

} // namespace fem
} // namespace femx
