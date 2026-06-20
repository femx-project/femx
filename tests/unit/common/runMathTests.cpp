#include <cmath>
#include <iostream>
#include <stdexcept>

#include <femx/core/Math.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class MathTests : public TestBase
{
public:
  TestOutcome vectorOps()
  {
    TestStatus status;
    status = true;

    const Vector<Real> x{1.0, -2.0, 3.0};
    const Vector<Real> y{4.0, 0.5, -1.0};

    status *= isEqual(dot(x, y), 0.0);
    status *= isEqual(squaredNorm(x), 14.0);
    status *= isEqual(norm(x), std::sqrt(14.0));
    status *= isEqual(rmse(x, y), std::sqrt(31.25 / 3.0));

    const Vector<Real> diff  = difference(x, y);
    status                  *= (diff.size() == 3);
    status                  *= isEqual(diff[0], -3.0);
    status                  *= isEqual(diff[1], -2.5);
    status                  *= isEqual(diff[2], 4.0);

    return status.report(__func__);
  }

  TestOutcome pointOps()
  {
    TestStatus status;
    status = true;

    const Point3 x{1.0, 2.0, 3.0};
    const Point3 y{4.0, -2.0, 1.0};

    status *= isEqual(dot(x, y), 3.0);
    status *= isEqual(squaredNorm(x), 14.0);
    status *= isEqual(norm(x), std::sqrt(14.0));
    status *= isEqual(sqDist(x, y), 29.0);
    status *= isEqual(distance(x, y), std::sqrt(29.0));

    const Point3 diff  = difference(x, y);
    status            *= isEqual(diff[0], -3.0);
    status            *= isEqual(diff[1], 4.0);
    status            *= isEqual(diff[2], 2.0);

    const Point3 cr  = cross(Point3{1.0, 0.0, 0.0}, Point3{0.0, 1.0, 0.0});
    status          *= isEqual(cr[0], 0.0);
    status          *= isEqual(cr[1], 0.0);
    status          *= isEqual(cr[2], 1.0);

    const Point3 dir  = unit(Point3{0.0, 3.0, 4.0});
    status           *= isEqual(dir[0], 0.0);
    status           *= isEqual(dir[1], 0.6);
    status           *= isEqual(dir[2], 0.8);

    status *= isEqual(triArea(Point3{0.0, 0.0, 0.0},
                              Point3{2.0, 0.0, 0.0},
                              Point3{0.0, 3.0, 0.0}),
                      3.0);
    status *= isEqual(radialSq(Point3{2.0, 3.0, 0.0},
                               Point3{1.0, 1.0, 0.0},
                               Point3{0.0, 2.0, 0.0}),
                      1.0);

    return status.report(__func__);
  }

  TestOutcome indexSetOps()
  {
    TestStatus status;
    status = true;

    Vector<Index> out{2, 4};
    appendUnique(out, Vector<Index>{4, 5, 2, 7});

    status *= (out.size() == 4);
    status *= (out[0] == 2);
    status *= (out[1] == 4);
    status *= (out[2] == 5);
    status *= (out[3] == 7);
    status *= contains(out, 5);
    status *= !contains(out, 9);

    appendUniqueExcept(out, Vector<Index>{1, 2, 6, 7}, Vector<Index>{1, 9});

    status *= (out.size() == 5);
    status *= (out[4] == 6);
    status *= !contains(out, 1);

    return status.report(__func__);
  }

  TestOutcome rejectsInvalidDimensions()
  {
    TestStatus status;
    status = true;

    bool threw = false;
    try
    {
      const Vector<Real> x{1.0};
      const Vector<Real> y{1.0, 2.0};
      (void) dot(x, y);
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }
    status *= threw;

    threw = false;
    try
    {
      (void) unit(Point3{0.0, 0.0, 0.0});
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }
    status *= threw;

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running math tests:\n";

  femx::tests::MathTests test;

  femx::tests::TestingResults result;
  result += test.vectorOps();
  result += test.pointOps();
  result += test.indexSetOps();
  result += test.rejectsInvalidDimensions();

  return result.summary();
}
