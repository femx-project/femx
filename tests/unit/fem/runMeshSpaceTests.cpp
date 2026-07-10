#include <array>
#include <cmath>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/ObservationGrid.hpp>
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

template <std::size_t N>
bool pointNear(const Point3& actual, const std::array<Real, N>& expected)
{
  static_assert(N == 3, "Point expectations must have three coordinates");
  for (std::size_t i = 0; i < N; ++i)
  {
    if (!near(actual[i], expected[i]))
    {
      return false;
    }
  }
  return true;
}

template <std::size_t N>
bool indexValuesEqual(const Vector<Index>&        actual,
                      const std::array<Index, N>& expected)
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

TestOutcome structuredQuadMesh()
{
  TestStatus status(__func__);

  const Mesh mesh = Mesh::makeStructuredQuad(2, 1, -1.0, 1.0, 2.0, 4.0);

  status *= mesh.dim() == 2;
  status *= mesh.numNodes() == 6;
  status *= mesh.numElems() == 2;

  status *= pointNear(mesh.node(0), std::array<Real, 3>{{-1.0, 2.0, 0.0}});
  status *= pointNear(mesh.node(1), std::array<Real, 3>{{0.0, 2.0, 0.0}});
  status *= pointNear(mesh.node(2), std::array<Real, 3>{{1.0, 2.0, 0.0}});
  status *= pointNear(mesh.node(3), std::array<Real, 3>{{-1.0, 4.0, 0.0}});
  status *= pointNear(mesh.node(5), std::array<Real, 3>{{1.0, 4.0, 0.0}});

  status *= mesh.elem(0).shape() == Element::Shape::Quadrilateral;
  status *= mesh.elem(1).shape() == Element::Shape::Quadrilateral;
  status *= indexValuesEqual(mesh.elem(0).nodeIds(),
                             std::array<Index, 4>{{0, 1, 4, 3}});
  status *= indexValuesEqual(mesh.elem(1).nodeIds(),
                             std::array<Index, 4>{{1, 2, 5, 4}});

  return status.report();
}

TestOutcome scalarFESpaceDofMap()
{
  TestStatus status(__func__);

  const Mesh        mesh = Mesh::makeStructuredQuad(2, 1);
  LagrangeQuadQ1    element;
  FESpace           space(&mesh, &element);
  space.setup();

  status *= space.numElems() == 2;
  status *= space.numComponents() == 1;
  status *= space.numShapesPerElem() == 4;
  status *= space.numDofsPerElem() == 4;
  status *= space.numDofs() == mesh.numNodes();
  status *= space.localDof(3, 0) == 3;
  status *= space.globalDof(4, 0) == 4;

  status *= indexValuesEqual(space.elemDofs(0),
                             std::array<Index, 4>{{0, 1, 4, 3}});
  status *= indexValuesEqual(space.elemDofs(1),
                             std::array<Index, 4>{{1, 2, 5, 4}});

  return status.report();
}

TestOutcome vectorFESpaceDofMap()
{
  TestStatus status(__func__);

  const Mesh        mesh = Mesh::makeStructuredQuad(1, 1);
  LagrangeQuadQ1    element;
  FESpace           space(&mesh, &element, 2);
  space.setup();

  status *= space.numElems() == 1;
  status *= space.numComponents() == 2;
  status *= space.numShapesPerElem() == 4;
  status *= space.numDofsPerElem() == 8;
  status *= space.numDofs() == 2 * mesh.numNodes();
  status *= space.localDof(0, 0) == 0;
  status *= space.localDof(0, 1) == 1;
  status *= space.localDof(3, 1) == 7;
  status *= space.globalDof(2, 0) == 4;
  status *= space.globalDof(2, 1) == 5;

  status *= indexValuesEqual(
      space.elemDofs(0),
      std::array<Index, 8>{{0, 1, 2, 3, 6, 7, 4, 5}});

  return status.report();
}

TestOutcome invalidFESpaceInputs()
{
  TestStatus status(__func__);

  const Mesh     mesh = Mesh::makeStructuredQuad(1, 1);
  LagrangeQuadQ1 element;

  bool threw = false;
  try
  {
    FESpace space(&mesh, &element, 0);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  Mesh triangle_like(2);
  triangle_like.addNode({0.0, 0.0, 0.0});
  triangle_like.addNode({1.0, 0.0, 0.0});
  triangle_like.addNode({0.0, 1.0, 0.0});
  triangle_like.addElem({0, 1, 2}, Element::Shape::Triangle, 2, 0, 0, {});

  threw = false;
  try
  {
    FESpace space(&triangle_like, &element);
    space.setup();
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome observationGridFromBounds()
{
  TestStatus status(__func__);

  const Vector<Point3> pts = fem::observationGridPoints(
      Point3{0.0, 10.0, -1.0},
      Point3{2.0, 20.0, 1.0},
      std::array<Index, 3>{{3, 2, 1}});

  status *= pts.size() == 6;
  status *= pointNear(pts[0], std::array<Real, 3>{{0.0, 10.0, 0.0}});
  status *= pointNear(pts[1], std::array<Real, 3>{{1.0, 10.0, 0.0}});
  status *= pointNear(pts[2], std::array<Real, 3>{{2.0, 10.0, 0.0}});
  status *= pointNear(pts[3], std::array<Real, 3>{{0.0, 20.0, 0.0}});
  status *= pointNear(pts[5], std::array<Real, 3>{{2.0, 20.0, 0.0}});

  const Vector<Point3> center = fem::observationGridPoints(
      Point3{-2.0, 2.0, 10.0},
      Point3{2.0, 6.0, 14.0},
      std::array<Index, 3>{{1, 1, 1}});

  status *= center.size() == 1;
  status *= pointNear(center[0], std::array<Real, 3>{{0.0, 4.0, 12.0}});

  return status.report();
}

TestOutcome observationGridFromSpacing()
{
  TestStatus status(__func__);

  const Vector<Point3> pts = fem::observationGridPoints(
      Point3{1.0, 2.0, 3.0},
      std::array<Index, 3>{{2, 1, 2}},
      Point3{0.5, 10.0, -1.0});

  status *= pts.size() == 4;
  status *= pointNear(pts[0], std::array<Real, 3>{{1.0, 2.0, 3.0}});
  status *= pointNear(pts[1], std::array<Real, 3>{{1.5, 2.0, 3.0}});
  status *= pointNear(pts[2], std::array<Real, 3>{{1.0, 2.0, 2.0}});
  status *= pointNear(pts[3], std::array<Real, 3>{{1.5, 2.0, 2.0}});

  bool threw = false;
  try
  {
    fem::observationGridPoints(Point3{0.0, 0.0, 0.0},
                               std::array<Index, 3>{{1, 0, 1}},
                               Point3{1.0, 1.0, 1.0});
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

  results += femx::tests::structuredQuadMesh();
  results += femx::tests::scalarFESpaceDofMap();
  results += femx::tests::vectorFESpaceDofMap();
  results += femx::tests::invalidFESpaceInputs();
  results += femx::tests::observationGridFromBounds();
  results += femx::tests::observationGridFromSpacing();

  return results.summary();
}
