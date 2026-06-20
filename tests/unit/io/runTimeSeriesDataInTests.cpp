#include <filesystem>
#include <iostream>

#include <femx/io/TimeSeriesDataIn.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/fem/Mesh.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class TimeSeriesDataInTests : public TestBase
{
public:
  TestOutcome readsTimeSeriesDataOut()
  {
    TestStatus status;
    status = true;

#ifndef FEMX_HAS_HDF5
    return status.report(__func__);
#else
    const auto dir =
        std::filesystem::temp_directory_path() / "femx_time_series_in_tests";
    std::filesystem::create_directories(dir);
    const auto basename = (dir / "series").string();

    Mesh mesh = Mesh::makeStructuredQuad(1, 1);
    Vector<Real> ux(mesh.numNodes());
    Vector<Real> uy(mesh.numNodes());
    Vector<Real> uz(mesh.numNodes());

    TimeSeriesDataOut out;
    out.attachMesh(mesh);
    for (Index step = 0; step < 2; ++step)
    {
      out.beginStep(0.25 + 0.5 * static_cast<Real>(step));
      for (Index node = 0; node < mesh.numNodes(); ++node)
      {
        ux[node] = static_cast<Real>(step) + mesh.node(node)[0];
        uy[node] = 2.0 * static_cast<Real>(step) + mesh.node(node)[1];
        uz[node] = 0.0;
      }
      out.addNodalVectorField("velocity", ux, uy, uz);
      out.addNodalVectorField("u_ref", uy, ux, uz);
    }
    out.write(basename);

    const TimeSeriesDataIn in = TimeSeriesDataIn::read(basename + ".xdmf");
    status *= (in.mesh().numNodes() == mesh.numNodes());
    status *= (in.mesh().numElems() == mesh.numElems());
    status *= (in.numSteps() == 2);
    status *= isEqual(in.time(0), 0.25);
    status *= isEqual(in.time(1), 0.75);

    const auto& velocity = in.vectorField(1, "velocity");
    status *= isEqual(velocity[0][3], 2.0);
    status *= isEqual(velocity[1][3], 3.0);
    status *= isEqual(velocity[2][3], 0.0);

    const auto& u_ref = in.vectorField(1, "u_ref");
    status *= isEqual(u_ref[0][3], 3.0);
    status *= isEqual(u_ref[1][3], 2.0);
    status *= isEqual(u_ref[2][3], 0.0);

    return status.report(__func__);
#endif
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running time series data input tests:\n";

  femx::tests::TimeSeriesDataInTests test;

  femx::tests::TestingResults result;
  result += test.readsTimeSeriesDataOut();

  return result.summary();
}
