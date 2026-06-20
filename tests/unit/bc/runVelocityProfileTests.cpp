#include <cmath>
#include <iostream>
#include <stdexcept>

#include <femx/fem/VelocityProfile.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class VelocityProfileTests : public TestBase
{
public:
  TestOutcome poiseuilleProfileUsesRadialDistance()
  {
    TestStatus status;
    status = true;

    const fem::AxialVelocityProfile profile = fem::poiseuilleProfile(
        Point3{0.0, 0.0, 0.0}, Point3{2.0, 0.0, 0.0}, 1.0);

    status *= isEqual(fem::profileFactor(profile, Point3{0.0, 0.0, 0.0}), 1.0);
    status *= isEqual(fem::profileFactor(profile, Point3{0.0, 0.5, 0.0}), 0.75);
    status *= isEqual(fem::profileFactor(profile, Point3{0.0, 2.0, 0.0}), 0.0);

    status *= isEqual(
        fem::velocityComponent(profile, Point3{0.0, 0.5, 0.0}, 2.0, 0),
        1.5);
    status *= isEqual(
        fem::velocityComponent(profile, Point3{0.0, 0.5, 0.0}, 2.0, 1),
        0.0);

    return status.report(__func__);
  }

  TestOutcome peakSpeedMapsQuantities()
  {
    TestStatus status;
    status = true;

    status *= isEqual(fem::peakSpeed("flowrate", "uniform", 6.0, 3.0), 2.0);
    status *= isEqual(fem::peakSpeed("mean_velocity", "uniform", 2.0), 2.0);
    status *= isEqual(
        fem::peakSpeed("flowrate", "poiseuille", 6.0, 3.0, 2.0),
        4.0);
    status *= isEqual(
        fem::peakSpeed("mean_velocity", "poiseuille", 2.0, 1.0, 1.5),
        3.0);
    status *= isEqual(
        fem::peakSpeed("max_velocity", "poiseuille", 2.0, 1.0, 1.5),
        2.0);

    return status.report(__func__);
  }

  TestOutcome boundaryCenterSupportsTagsAndNames()
  {
    TestStatus status;
    status = true;

    Mesh mesh(3);
    mesh.addNode({0.0, 0.0, 0.0});
    mesh.addNode({0.0, 2.0, 0.0});
    mesh.addNode({0.0, 0.0, 2.0});
    mesh.addNode({3.0, 0.0, 0.0});

    Mesh::BoundaryFacet tri;
    tri.dim           = 2;
    tri.physical_tag  = 7;
    tri.physical_name = "inlet";
    tri.shape         = Cell::Shape::Triangle;
    tri.node_ids      = {0, 1, 2};
    mesh.addBoundaryFacet(tri);

    Mesh::BoundaryFacet line;
    line.dim           = 1;
    line.physical_tag  = 8;
    line.physical_name = "edge";
    line.shape         = Cell::Shape::Segment;
    line.node_ids      = {0, 3};
    mesh.addBoundaryFacet(line);

    const Point3 tri_center  = fem::boundaryCenter(mesh, 7);
    status                  *= isEqual(tri_center[0], 0.0);
    status                  *= isEqual(tri_center[1], 2.0 / 3.0);
    status                  *= isEqual(tri_center[2], 2.0 / 3.0);

    const Point3 line_center  = fem::boundaryCenter(mesh, "edge");
    status                   *= isEqual(line_center[0], 1.5);
    status                   *= isEqual(line_center[1], 0.0);
    status                   *= isEqual(line_center[2], 0.0);

    return status.report(__func__);
  }

  TestOutcome sinePulseAndValidation()
  {
    TestStatus status;
    status = true;

    status *= isEqual(fem::sinePulseFactor(0.25, 0.5, 1.0), 1.5);

    bool threw = false;
    try
    {
      (void) fem::poiseuilleProfile(
          Point3{0.0, 0.0, 0.0}, Point3{1.0, 0.0, 0.0}, 0.0);
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }
    status *= threw;

    threw = false;
    try
    {
      (void) fem::sinePulseFactor(0.0, 0.5, 0.0);
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
  std::cout << "Running velocity profile tests:\n";

  femx::tests::VelocityProfileTests test;

  femx::tests::TestingResults result;
  result += test.poiseuilleProfileUsesRadialDistance();
  result += test.peakSpeedMapsQuantities();
  result += test.boundaryCenterSupportsTagsAndNames();
  result += test.sinePulseAndValidation();

  return result.summary();
}
