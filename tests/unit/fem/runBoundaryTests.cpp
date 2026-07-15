#include <array>
#include <cmath>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/fem/BoundarySurface.hpp>
#include <femx/fem/DirichletBC.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/TimeDirichletData.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>

namespace femx
{
using namespace fem;

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

Mesh makeBoundaryLineMesh()
{
  Mesh mesh(2);
  mesh.addNode({0.0, 0.0, 0.0});
  mesh.addNode({0.0, 0.5, 0.0});
  mesh.addNode({0.0, 1.0, 0.0});
  mesh.addPhysicalName(1, 4, "inlet");
  mesh.addBoundaryFacet({1,
                         1,
                         4,
                         "inlet",
                         Element::Shape::Segment,
                         Vector<Index>{0, 1}});
  mesh.addBoundaryFacet({1,
                         2,
                         4,
                         "inlet",
                         Element::Shape::Segment,
                         Vector<Index>{1, 2}});
  return mesh;
}

Mesh makeBoundaryTriangleMesh()
{
  Mesh mesh(3);
  mesh.addNode({0.0, 0.0, 0.0});
  mesh.addNode({1.0, 0.0, 0.0});
  mesh.addNode({1.0, 1.0, 0.0});
  mesh.addNode({0.0, 1.0, 0.0});
  mesh.addNode({0.5, 0.5, 0.0});
  mesh.addPhysicalName(2, 4, "inlet");
  const Vector<Vector<Index>> facets = {
      {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
  for (Index i = 0; i < facets.size(); ++i)
  {
    mesh.addBoundaryFacet({2,
                           i + 1,
                           4,
                           "inlet",
                           Element::Shape::Triangle,
                           facets[i]});
  }
  return mesh;
}

Real sumEntries(const SparseTripletMatrix& matrix)
{
  Real out = 0.0;
  for (Real value : matrix.values)
  {
    out += value;
  }
  return out;
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

TestOutcome boundarySurfaceLineMatrices()
{
  TestStatus status(__func__);

  const Mesh            mesh = makeBoundaryLineMesh();
  const BoundarySurface surface(mesh, "inlet");
  const auto            matrices = surface.scalarMatrices();

  status *= surface.dim() == 1;
  status *= surface.numNodes() == 3;
  status *= surface.numElements() == 2;
  status *= valuesEqual(surface.meshNodeIds(),
                        std::array<Index, 3>{{0, 1, 2}});
  status *= valuesEqual(surface.rimNodeIds(),
                        std::array<Index, 2>{{0, 2}});
  status *= valuesNear(matrices.load,
                       std::array<Real, 3>{{0.25, 0.5, 0.25}});
  status *= near(sumEntries(matrices.mass), 1.0);
  status *= near(sumEntries(matrices.stiffness), 0.0);

  const BoundarySurface by_tag(mesh, 4);
  status *= by_tag.numNodes() == surface.numNodes();

  return status.report();
}

TestOutcome boundarySurfaceTriangleMatrices()
{
  TestStatus status(__func__);

  const Mesh            mesh = makeBoundaryTriangleMesh();
  const BoundarySurface surface(mesh, 4);
  const auto            matrices = surface.scalarMatrices();

  status *= surface.dim() == 2;
  status *= surface.numNodes() == 5;
  status *= surface.numElements() == 4;
  status *= surface.rimNodeIds().size() == 4;
  status *= near(sumEntries(matrices.mass), 1.0);
  status *= near(sumEntries(matrices.stiffness), 0.0);

  Real load_sum = 0.0;
  for (Real value : matrices.load)
  {
    load_sum += value;
  }
  status *= near(load_sum, 1.0);

  return status.report();
}

TestOutcome dirichletConditionFromPhysicalTag()
{
  TestStatus status(__func__);

  const Mesh     mesh = makeBoundaryMesh();
  LagrangeQuadQ1 element;
  FESpace        space(&mesh, &element, 2);
  space.setup();

  DirichletBC bc;
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
  status *= valuesNear(bc.values(), std::array<Real, 2>{{0.5, 1.5}});

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

  DirichletBC bc;
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
  status *= valuesNear(bc.values(), std::array<Real, 2>{{12.0, 13.0}});

  return status.report();
}

TestOutcome dirichletControlBasics()
{
  TestStatus status(__func__);

  const DirichletControl control(Vector<Index>{3, 5, 9});
  status *= control.numStateDofs() == 3;
  status *= control.numControlParams() == 3;
  status *= control.stateDof(1) == 5;
  status *= valuesEqual(control.stateDofs(),
                        std::array<Index, 3>{{3, 5, 9}});

  Vector<Real> mapped;
  control.apply(Vector<Real>{1.0, 2.0, 3.0}, mapped);
  status *= valuesNear(mapped, std::array<Real, 3>{{1.0, 2.0, 3.0}});

  Vector<Real> transpose;
  control.applyTranspose(Vector<Real>{4.0, 5.0, 6.0}, transpose);
  status *= valuesNear(transpose,
                       std::array<Real, 3>{{4.0, 5.0, 6.0}});

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

TestOutcome mappedDirichletControl()
{
  TestStatus status(__func__);

  const DirichletControl control(
      Vector<Index>{4, 7, 8},
      2,
      Vector<DirichletControlMapEntry>{{0, 0, 2.0},
                                       {1, 0, -1.0},
                                       {1, 1, 3.0},
                                       {2, 1, 0.5}});

  status *= control.numStateDofs() == 3;
  status *= control.numControlParams() == 2;

  const Vector<Real> parameters{2.0, -1.0};
  Vector<Real>       linear;
  control.apply(parameters, linear);
  status *= valuesNear(linear,
                       std::array<Real, 3>{{4.0, -5.0, -0.5}});

  const Vector<Real> state_direction{1.0, 2.0, -4.0};
  Vector<Real>       transpose;
  control.applyTranspose(state_direction, transpose);
  status *= valuesNear(transpose, std::array<Real, 2>{{0.0, 4.0}});

  Real state_dot = 0.0;
  for (Index i = 0; i < linear.size(); ++i)
  {
    state_dot += linear[i] * state_direction[i];
  }
  Real ctr_dot = 0.0;
  for (Index i = 0; i < parameters.size(); ++i)
  {
    ctr_dot += parameters[i] * transpose[i];
  }
  status *= near(state_dot, ctr_dot);

  bool threw = false;
  try
  {
    DirichletControl duplicate_entry(
        Vector<Index>{0},
        1,
        Vector<DirichletControlMapEntry>{{0, 0, 1.0},
                                         {0, 0, 2.0}});
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

TestOutcome normalVelocityControlMapping()
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

  const DirichletControl control =
      makeNormalVelocityControl(mixed, "right", Vector<Real>{3.0, 4.0});
  status *= valuesEqual(control.stateDofs(),
                        std::array<Index, 4>{{2, 3, 6, 7}});
  status *= control.numControlParams() == 2;

  Vector<Real> mapped;
  control.apply(Vector<Real>{10.0, 20.0}, mapped);
  status *= valuesNear(mapped,
                       std::array<Real, 4>{{6.0, 8.0, 12.0, 16.0}});

  const DirichletControl active =
      control.withoutStateDofs(Vector<Index>{2, 3});
  status *= valuesEqual(active.stateDofs(),
                        std::array<Index, 2>{{6, 7}});
  status *= active.numControlParams() == 1;
  active.apply(Vector<Real>{5.0}, mapped);
  status *= valuesNear(mapped, std::array<Real, 2>{{3.0, 4.0}});

  const DirichletControl axial =
      makeNormalVelocityControl(mixed, "right", Vector<Real>{1.0, 0.0});
  axial.apply(Vector<Real>{2.0, 3.0}, mapped);
  status *= valuesNear(mapped,
                       std::array<Real, 4>{{2.0, 0.0, 3.0, 0.0}});

  return status.report();
}

TestOutcome timeDirichletValueCompilation()
{
  TestStatus status(__func__);

  const TimeDirichletData values = makeTimeDirichletData(
      5,
      2,
      0.5,
      [](Real time)
      {
        DirichletBC condition;
        condition.addDof(1, 1.0 + time);
        condition.addDof(3, -time);
        return condition;
      });

  status *= valuesEqual(values.dofs, std::array<Index, 2>{{1, 3}});
  status *= valuesNear(
      values.values, std::array<Real, 4>{{1.5, -0.5, 2.0, -1.0}});
  status *= valuesNear(
      values.initial_state,
      std::array<Real, 5>{{0.0, 1.0, 0.0, 0.0, 0.0}});

  bool threw = false;
  try
  {
    makeTimeDirichletData(
        2,
        1,
        1.0,
        [](Real)
        {
          DirichletBC condition;
          condition.addDof(0, 1.0);
          condition.addDof(0, 2.0);
          return condition;
        });
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
  results += femx::tests::boundarySurfaceLineMatrices();
  results += femx::tests::boundarySurfaceTriangleMatrices();
  results += femx::tests::dirichletConditionFromPhysicalTag();
  results += femx::tests::dirichletConditionFromMarker();
  results += femx::tests::dirichletControlBasics();
  results += femx::tests::mappedDirichletControl();
  results += femx::tests::velocityControlFromMixedSpace();
  results += femx::tests::normalVelocityControlMapping();
  results += femx::tests::timeDirichletValueCompilation();

  return results.summary();
}
