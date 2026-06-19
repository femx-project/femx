#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "ObservationVti.hpp"
#include <femx/inverse/ObservationGrid.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class MakeObsObservationVtiTests : public TestBase
{
public:
  TestOutcome writeObservationTimeSeries()
  {
    TestStatus status;
    status = true;

    const auto dir =
        std::filesystem::temp_directory_path() / "femx_obs_vti_tests";
    std::filesystem::create_directories(dir);

    make_obs::Params prm;
    prm.output.write_vti    = true;
    prm.output.vti_basename = (dir / "obs-grid").string();
    prm.obs.type            = "grid";
    prm.obs.components      = {0, 1};
    prm.obs.grid            = navier_var::ObservationParams::Grid{};
    prm.obs.grid->lower     = {0.0, 0.0, 0.0};
    prm.obs.grid->upper     = {1.0, 1.0, 0.0};
    prm.obs.grid->counts    = {2, 2, 1};

    const auto points = inverse::cartesianObsPoints(
        Point3{0.0, 0.0, 0.0}, Point3{1.0, 1.0, 0.0},
        prm.obs.grid->counts);
    Vector<Index> components{0, 1};
    inverse::TimeObservationData data(2, 8);
    data.setLayout("point", points, components);
    data.setTimeValues(Vector<Real>{1.0, 1.5});
    for (Index level = 0; level < data.numLevels(); ++level)
    {
      Vector<Real> values = data[level];
      for (Index i = 0; i < values.size(); ++i)
      {
        values[i] = 10.0 * static_cast<Real>(level)
                    + static_cast<Real>(i);
      }
    }

    const std::string pvd = make_obs::writeObservationVtiOutputs(prm, data);
    const auto        vti0 = dir / "obs-grid_00000.vti";
    const auto        vti1 = dir / "obs-grid_00001.vti";
    status                *= pvd == (dir / "obs-grid.pvd").string();
    status                *= std::filesystem::exists(pvd);
    status                *= std::filesystem::exists(vti0);
    status                *= std::filesystem::exists(vti1);

    std::ifstream pvd_in(pvd);
    const std::string pvd_text((std::istreambuf_iterator<char>(pvd_in)),
                               std::istreambuf_iterator<char>());
    status *= pvd_text.find("timestep=\"1\"") != std::string::npos;
    status *= pvd_text.find("obs-grid_00001.vti") != std::string::npos;

    std::ifstream vti_in(vti0, std::ios::binary);
    const std::string vti_text((std::istreambuf_iterator<char>(vti_in)),
                               std::istreambuf_iterator<char>());
    status *= vti_text.find("CellData") != std::string::npos;
    status *= vti_text.find("Name=\"velocity\"") != std::string::npos;
    status *= vti_text.find("WholeExtent=\"0 2 0 2 0 0\"")
              != std::string::npos;

    const auto marker = vti_text.find("\n_");
    if (marker != std::string::npos
        && marker + 2 + sizeof(std::uint64_t) <= vti_text.size())
    {
      std::uint64_t bytes = 0;
      std::memcpy(&bytes, vti_text.data() + marker + 2, sizeof(bytes));
      status *= bytes == static_cast<std::uint64_t>(4 * 3) * sizeof(Real);
    }
    else
    {
      status = false;
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running make-obs observation VTI tests:\n";

  femx::tests::MakeObsObservationVtiTests test;

  femx::tests::TestingResults result;
  result += test.writeObservationTimeSeries();

  return result.summary();
}
