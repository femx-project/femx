#include <array>
#include <cmath>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/fem/BoundaryCondition.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>

namespace femx
{
namespace tests
{
namespace
{

bool near(Real a, Real b)
{
  return std::abs(a - b) <= 1.0e-12;
}

template <class T, std::size_t N>
bool valuesEqual(const Vector<T>& actual, const std::array<T, N>& expected)
{
  if (actual.size() != static_cast<Index>(N))
  {
    return false;
  }
  for (std::size_t i = 0; i < N; ++i)
  {
    if (actual[static_cast<Index>(i)] != expected[i])
    {
      return false;
    }
  }
  return true;
}

template <std::size_t N>
bool valuesNear(const Vector<Real>&        actual,
                const std::array<Real, N>& expected)
{
  if (actual.size() != static_cast<Index>(N))
  {
    return false;
  }
  for (std::size_t i = 0; i < N; ++i)
  {
    if (!near(actual[static_cast<Index>(i)], expected[i]))
    {
      return false;
    }
  }
  return true;
}

Mesh makeBoundaryMesh()
{
  Mesh mesh = Mesh::makeStructuredQuad(1, 1);
  mesh.addPhysicalName(1, 7, "left");
  mesh.addPhysicalName(1, 8, "right");
  mesh.addBoundaryFacet({1,
                         1,
                         7,
                         "left",
                         Element::Shape::Segment,
                         Vector<Index>{0, 2}});
  mesh.addBoundaryFacet({1,
                         2,
                         8,
                         "right",
                         Element::Shape::Segment,
                         Vector<Index>{1, 3}});
  return mesh;
}

TestOutcome boundaryFacetLookup()
{
  TestStatus status(__func__);

  const Mesh mesh = makeBoundaryMesh();

  status *= mesh.physicalName(1, 7) == "left";
  status *= mesh.physicalName(1, 8) == "right";
  status *= mesh.physicalName(1, 99).empty();

  const Vector<Mesh::BoundaryFacet> left  = mesh.boundaryFacets("left");
  status                                 *= left.size() == 1;
  status                                 *= left[0].ptag == 7;
  status                                 *= left[0].pname == "left";
  status                                 *= valuesEqual(left[0].nids, std::array<Index, 2>{{0, 2}});

  return status.report();
}

TestOutcome dirichletConditionFromPhysicalTag()
{
  TestStatus status(__func__);

  const Mesh     mesh = makeBoundaryMesh();
  LagrangeQuadQ1 element;
  FESpace        space(&mesh, &element, 2);
  space.setup();

  DirichletCondition bc;
  bc.addBoundary(
      space,
      7,
      [](const Mesh::Node& point, Real time)
      {
        return point[0] + point[1] + time;
      },
      0.5,
      1);

  status *= valuesEqual(bc.dofs(), std::array<Index, 2>{{1, 5}});
  status *= valuesNear(bc.vals(), std::array<Real, 2>{{0.5, 1.5}});

  bool threw = false;
  try
  {
    bc.addBoundary(space, 99, 1.0);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome dirichletConditionFromMarker()
{
  TestStatus status(__func__);

  const Mesh     mesh = makeBoundaryMesh();
  LagrangeQuadQ1 element;
  FESpace        space(&mesh, &element, 2);
  space.setup();

  DirichletCondition bc;
  bc.addBoundary(
      space,
      [](const Mesh::Node& point, Real)
      {
        return near(point[0], 1.0);
      },
      [](const Mesh::Node& point, Real time)
      {
        return 10.0 * point[0] + point[1] + time;
      },
      2.0,
      0);

  status *= valuesEqual(bc.dofs(), std::array<Index, 2>{{2, 6}});
  status *= valuesNear(bc.vals(), std::array<Real, 2>{{12.0, 13.0}});

  return status.report();
}

TestOutcome dirichletControlBasics()
{
  TestStatus status(__func__);

  const DirichletControl control(Vector<Index>{3, 5, 9});
  status *= control.numDofs() == 3;
  status *= control.numParams(4) == 12;
  status *= control.stateDof(1) == 5;
  status *= control.paramIndex(2, 1) == 7;
  status *= valuesEqual(control.stateDofs(),
                        std::array<Index, 3>{{3, 5, 9}});

  bool threw = false;
  try
  {
    DirichletControl duplicate(Vector<Index>{1, 1});
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  threw = false;
  try
  {
    control.stateDof(3);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome velocityControlFromMixedSpace()
{
  TestStatus status(__func__);

  const Mesh     mesh = makeBoundaryMesh();
  LagrangeQuadQ1 element;
  FESpace        velocity(&mesh, &element, 2);
  FESpace        pressure(&mesh, &element, 1);

  MixedFESpace mixed;
  mixed.addField(velocity);
  mixed.addField(pressure);
  mixed.setup();

  const DirichletControl by_tag  = makeVelocityControl(mixed, 8);
  const DirichletControl by_name = makeVelocityControl(mixed, "right");

  status *= valuesEqual(by_tag.stateDofs(),
                        std::array<Index, 4>{{2, 3, 6, 7}});
  status *= valuesEqual(by_name.stateDofs(),
                        std::array<Index, 4>{{2, 3, 6, 7}});

  bool threw = false;
  try
  {
    makeVelocityControl(mixed, "missing");
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::boundaryFacetLookup();
  results += femx::tests::dirichletConditionFromPhysicalTag();
  results += femx::tests::dirichletConditionFromMarker();
  results += femx::tests::dirichletControlBasics();
  results += femx::tests::velocityControlFromMixedSpace();

  return results.summary();
}
