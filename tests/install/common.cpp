#include <cmath>
#include <stdexcept>

#include <femx/common/Math.hpp>

int main()
{
  const femx::Point3                 a{0.0, 0.0, 0.0};
  const femx::Point3                 b{1.0, 2.0, 2.0};
  const femx::HostVector<femx::Real> values{3.0, 4.0};

  if (std::abs(femx::distance(a, b) - 3.0) > 1.0e-12)
  {
    throw std::runtime_error("Point3 distance mismatch");
  }
  if (std::abs(femx::norm(values) - 5.0) > 1.0e-12)
  {
    throw std::runtime_error("HostVector norm mismatch");
  }
  return 0;
}
