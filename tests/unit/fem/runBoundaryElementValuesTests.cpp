#include <iostream>

#include <femx/fem/BoundaryElementValues.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class BoundaryElementValuesTests : public TestBase
{
public:
  TestOutcome segmentValues()
  {
    TestStatus status;
    status = true;

    Mesh mesh(2);
    mesh.addNode({0.0, 0.0, 0.0});
    mesh.addNode({2.0, 0.0, 0.0});

    Mesh::BoundaryFacet facet;
    facet.shape    = Cell::Shape::Segment;
    facet.node_ids = {0, 1};

    const auto            quad = GaussQuadrature::segment(2);
    BoundaryElementValues values(quad);
    values.reinit(mesh, facet);

    status *= (values.numNodes() == 2);
    status *= (values.dim() == 2);
    status *= (values.numQuadraturePoints() == 2);

    Real length = 0.0;
    for (Index iq = 0; iq < values.numQuadraturePoints(); ++iq)
    {
      const auto N  = values.N(iq);
      const auto x  = values.point(iq);
      const auto n  = values.normal(iq);
      status       *= isEqual(N[0] + N[1], 1.0);
      status       *= isEqual(x[0], 1.0 + quad[iq][0]);
      status       *= isEqual(x[1], 0.0);
      status       *= isEqual(n[0], 0.0);
      status       *= isEqual(n[1], -1.0);
      length       += values.JxW(iq);
    }

    status *= isEqual(length, 2.0);

    return status.report(__func__);
  }

  TestOutcome triangleValues()
  {
    TestStatus status;
    status = true;

    Mesh mesh(3);
    mesh.addNode({0.0, 0.0, 0.0});
    mesh.addNode({2.0, 0.0, 0.0});
    mesh.addNode({0.0, 3.0, 0.0});

    Mesh::BoundaryFacet facet;
    facet.shape    = Cell::Shape::Triangle;
    facet.node_ids = {0, 1, 2};

    const auto            quad = GaussQuadrature::triangle(1);
    BoundaryElementValues values(quad);
    values.reinit(mesh, facet);

    status *= (values.numNodes() == 3);
    status *= (values.dim() == 3);
    status *= (values.numQuadraturePoints() == 1);

    const auto N  = values.N(0);
    const auto x  = values.point(0);
    const auto n  = values.normal(0);
    status       *= isEqual(N[0], 1.0 / 3.0);
    status       *= isEqual(N[1], 1.0 / 3.0);
    status       *= isEqual(N[2], 1.0 / 3.0);
    status       *= isEqual(x[0], 2.0 / 3.0);
    status       *= isEqual(x[1], 1.0);
    status       *= isEqual(x[2], 0.0);
    status       *= isEqual(n[0], 0.0);
    status       *= isEqual(n[1], 0.0);
    status       *= isEqual(n[2], 1.0);
    status       *= isEqual(values.JxW(0), 3.0);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running boundary element values tests:\n";

  femx::tests::BoundaryElementValuesTests test;

  femx::tests::TestingResults result;
  result += test.segmentValues();
  result += test.triangleValues();

  return result.summary();
}
