#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <femx/core/Math.hpp>
#include <femx/solve/TimeStateTrajectory.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/fem/ObservationGrid.hpp>
#include <femx/problem/TimeObservationData.hpp>
#include <femx/fem/TimePointSampler.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/fem/Mesh.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

struct QuadFixture
{
  Mesh           mesh;
  LagrangeQuadQ1 elem;
  FESpace        vel;
  MixedFESpace   space;

  QuadFixture()
    : mesh(Mesh::makeStructuredQuad(2, 2, 0.0, 1.0, 0.0, 1.0)),
      vel(&mesh, &elem, 2)
  {
    space.addField(vel);
    space.setup();
  }

  Real value(Index         comp,
             const Point3& point,
             Real          bias = 0.0) const
  {
    if (comp == 0)
    {
      return bias + 1.0 + 2.0 * point[0] + 3.0 * point[1];
    }
    return bias - 2.0 - point[0] + 4.0 * point[1];
  }

  void fillState(Vector<Real>& state,
                 Real          bias = 0.0) const
  {
    state.resize(space.numDofs());
    const auto field = space.field(0);
    for (Index node = 0; node < mesh.numNodes(); ++node)
    {
      const Point3 point = mesh.node(node);
      for (Index comp = 0; comp < field.numComponents(); ++comp)
      {
        state[field.globalDof(node, comp)] = value(comp, point, bias);
      }
    }
  }
};

class TimePointSamplerTests : public TestBase
{
public:
  TestOutcome observationGridPointsBuildSparseGrid()
  {
    TestStatus status;
    status = true;

    const auto points = fem::observationGridPoints(
        Point3{0.0, 0.0, 0.0},
        Point3{1.0, 2.0, 0.0},
        std::array<Index, 3>{3, 2, 1});

    status *= (points.size() == 6);
    status *= isEqual(points[0][0], 0.0);
    status *= isEqual(points[0][1], 0.0);
    status *= isEqual(points[1][0], 0.5);
    status *= isEqual(points[2][0], 1.0);
    status *= isEqual(points[3][0], 0.0);
    status *= isEqual(points[3][1], 2.0);

    const auto spaced = fem::observationGridPoints(
        Point3{1.0, 2.0, 3.0},
        std::array<Index, 3>{2, 2, 2},
        Point3{0.5, 1.0, 2.0});

    status *= (spaced.size() == 8);
    status *= isEqual(spaced.back()[0], 1.5);
    status *= isEqual(spaced.back()[1], 3.0);
    status *= isEqual(spaced.back()[2], 5.0);

    return status.report(__func__);
  }

  TestOutcome samplesQuadFieldAtPhysicalPoints()
  {
    TestStatus status;
    status = true;

    QuadFixture               fixture;
    const std::vector<Point3> points{
        Point3{0.25, 0.50, 0.0},
        Point3{0.75, 0.25, 0.0}};
    const fem::TimePointSampler sampler(
        2,
        fixture.space,
        0,
        points,
        Vector<Index>{0, 1},
        3);

    Vector<Real> state;
    fixture.fillState(state);
    Vector<Real> prm(3);
    Vector<Real> obs;
    sampler.observe(1, state, prm, obs);

    status *= (obs.size() == 4);
    status *= isEqual(obs[0], fixture.value(0, points[0]));
    status *= isEqual(obs[1], fixture.value(1, points[0]));
    status *= isEqual(obs[2], fixture.value(0, points[1]));
    status *= isEqual(obs[3], fixture.value(1, points[1]));

    return status.report(__func__);
  }

  TestOutcome stateJacobianTransposeIsAdjoint()
  {
    TestStatus status;
    status = true;

    QuadFixture fixture;
    const auto  points = fem::observationGridPoints(
        Point3{0.25, 0.25, 0.0},
        Point3{0.75, 0.75, 0.0},
        std::array<Index, 3>{2, 2, 1});
    const fem::TimePointSampler sampler(
        1,
        fixture.space,
        0,
        points,
        Vector<Index>{0, 1},
        2);

    Vector<Real> state;
    fixture.fillState(state);
    Vector<Real> prm(2);

    Vector<Real> dir(fixture.space.numDofs());
    for (Index i = 0; i < dir.size(); ++i)
    {
      dir[i] = 0.25 + static_cast<Real>(i);
    }

    Vector<Real> seed(sampler.numObservations());
    for (Index i = 0; i < seed.size(); ++i)
    {
      seed[i] = -0.5 + 0.3 * static_cast<Real>(i);
    }

    Vector<Real> obs_dir;
    sampler.applyStateJac(0, state, prm, dir, obs_dir);

    Vector<Real> state_grad;
    sampler.applyStateJacT(0, state, prm, seed, state_grad);

    status *= isEqual(dot(obs_dir, seed), dot(dir, state_grad));

    return status.report(__func__);
  }

  TestOutcome samplesReferenceTrajectoryThroughObs()
  {
    TestStatus status;
    status = true;

    QuadFixture                     fixture;
    const std::vector<Point3>       points{Point3{0.25, 0.50, 0.0}};
    const fem::TimePointSampler obs(
        2,
        fixture.space,
        0,
        points,
        Vector<Index>{0, 1},
        1);

    solve::TimeStateTrajectory tr(2, fixture.space.numDofs());
    Vector<Real>            level0 = tr[0];
    Vector<Real>            level1 = tr[1];
    Vector<Real>            level2 = tr[2];
    fixture.fillState(level0, 0.0);
    fixture.fillState(level1, 10.0);
    fixture.fillState(level2, 20.0);

    Vector<Real> prm(1);
    const auto   data = problem::sampleTimeObs(obs, tr, prm);

    status *= (data.size() == 3);
    status *= isEqual(data[1][0], fixture.value(0, points[0], 10.0));
    status *= isEqual(data[2][1], fixture.value(1, points[0], 20.0));

    return status.report(__func__);
  }

  TestOutcome samplesTetraFieldAtPhysicalPoint()
  {
    TestStatus status;
    status = true;

    Mesh mesh(3);
    mesh.addNode({0.0, 0.0, 0.0});
    mesh.addNode({1.0, 0.0, 0.0});
    mesh.addNode({0.0, 1.0, 0.0});
    mesh.addNode({0.0, 0.0, 1.0});
    mesh.addCell({0, 1, 2, 3}, Cell::Shape::Tetrahedron, 3, 0, 0, {});

    LagrangeTetrahedronP1 elem;
    FESpace               vel(&mesh, &elem, 3);
    MixedFESpace          space;
    space.addField(vel);
    space.setup();

    const auto   field = space.field(0);
    Vector<Real> state(space.numDofs());
    for (Index node = 0; node < mesh.numNodes(); ++node)
    {
      const Point3 point              = mesh.node(node);
      state[field.globalDof(node, 0)] = point[0] + point[1] + point[2];
      state[field.globalDof(node, 1)] = 2.0 * point[0] - point[1];
      state[field.globalDof(node, 2)] = 1.0 + point[2];
    }

    const std::vector<Point3>       points{Point3{0.2, 0.3, 0.1}};
    const fem::TimePointSampler sampler(
        1,
        space,
        0,
        points,
        Vector<Index>{0, 2},
        0);

    Vector<Real> prm;
    Vector<Real> obs;
    sampler.observe(0, state, prm, obs);

    status *= (obs.size() == 2);
    status *= isEqual(obs[0], 0.6);
    status *= isEqual(obs[1], 1.1);

    return status.report(__func__);
  }

  TestOutcome rejectsPointOutsideMesh()
  {
    TestStatus status;
    status = true;

    bool threw = false;
    try
    {
      QuadFixture                     fixture;
      const fem::TimePointSampler sampler(
          1,
          fixture.space,
          0,
          std::vector<Point3>{Point3{2.0, 0.5, 0.0}},
          Vector<Index>{0},
          0);
      (void) sampler;
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }

    status *= threw;
    return status.report(__func__);
  }

  TestOutcome filtersPointsInsideMesh()
  {
    TestStatus status;
    status = true;

    QuadFixture fixture;
    const std::vector<Point3> points{
        Point3{0.25, 0.25, 0.0},
        Point3{2.00, 0.50, 0.0},
        Point3{1.00, 1.00, 0.0},
        Point3{0.50, -0.5, 0.0}};

    const auto filtered =
        fem::TimePointSampler::filterPointsInside(
            fixture.space, 0, points);

    status *= fem::TimePointSampler::containsPoint(
        fixture.space, 0, Point3{0.50, 0.50, 0.0});
    status *= !fem::TimePointSampler::containsPoint(
        fixture.space, 0, Point3{1.50, 0.50, 0.0});
    status *= (filtered.size() == 2);
    status *= isEqual(filtered[0][0], 0.25);
    status *= isEqual(filtered[0][1], 0.25);
    status *= isEqual(filtered[1][0], 1.0);
    status *= isEqual(filtered[1][1], 1.0);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running time point sampler tests:\n";

  femx::tests::TimePointSamplerTests test;

  femx::tests::TestingResults result;
  result += test.observationGridPointsBuildSparseGrid();
  result += test.samplesQuadFieldAtPhysicalPoints();
  result += test.stateJacobianTransposeIsAdjoint();
  result += test.samplesReferenceTrajectoryThroughObs();
  result += test.samplesTetraFieldAtPhysicalPoint();
  result += test.rejectsPointOutsideMesh();
  result += test.filtersPointsInsideMesh();

  return result.summary();
}
