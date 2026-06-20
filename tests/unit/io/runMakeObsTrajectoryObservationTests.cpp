#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "TrajectoryObservation.hpp"
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/io/TimeSeriesDataIn.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class MakeObsTrajectoryObservationTests : public TestBase
{
public:
  TestOutcome samplesObservationFromTrajectory()
  {
    TestStatus status;
    status = true;

#ifndef FEMX_HAS_HDF5
    return status.report(__func__);
#else
    const auto dir =
        std::filesystem::temp_directory_path() / "femx_make_obs_trajectory_tests";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    Mesh mesh = Mesh::makeStructuredQuad(1, 1);

    Vector<Real> ux(mesh.numNodes());
    Vector<Real> uy(mesh.numNodes());
    Vector<Real> uz(mesh.numNodes());
    uz.setZero();

    TimeSeriesDataOut out;
    out.attachMesh(mesh);
    for (Index step = 0; step < 2; ++step)
    {
      out.beginStep(static_cast<Real>(step));
      for (Index node = 0; node < mesh.numNodes(); ++node)
      {
        ux[node] = mesh.node(node)[0] + 2.0 * static_cast<Real>(step);
        uy[node] = mesh.node(node)[1] + 4.0 * static_cast<Real>(step);
      }
      out.addNodalVectorField("velocity", ux, uy, uz);
    }

    const auto trajectory = dir / "trajectory";
    out.write(trajectory.string());

    make_obs::Params prm;
    prm.input.trajectory       = trajectory.string() + ".xdmf";
    prm.output.file            = (dir / "obs.txt").string();
    prm.output.write_vti       = true;
    prm.output.vti_basename    = (dir / "obs-grid").string();
    prm.output.write_reference = true;
    prm.output.reference_basename =
        (dir / "obs-reference").string();
    prm.obs.type       = "grid";
    prm.obs.components = {0, 1};
    prm.obs.grid       = navier_var::ObservationParams::Grid{};
    prm.obs.grid->lower  = {0.0, 0.0, 0.0};
    prm.obs.grid->upper  = {1.0, 1.0, 0.0};
    prm.obs.grid->counts = {2, 2, 1};
    prm.time.start_time = 0.5;
    prm.time.end_time   = 0.5;
    prm.time.num_points = 1;

    make_obs::writeTrajectoryObservationOutputs(prm);

    const inverse::TimeObservationData data =
        inverse::readTimeObsData(prm.output.file);
    status *= (data.numLevels() == 1);
    status *= (data.numObservations() == 8);
    status *= data.hasLayout();
    status *= data.hasTimeValues();
    status *= isEqual(data.timeValue(0), 0.0);

    const Vector<Real> values = data[0];
    for (Index point = 0; point < static_cast<Index>(data.points().size());
         ++point)
    {
      const Point3& p = data.points()[static_cast<std::size_t>(point)];
      status *= isEqual(values[2 * point], p[0] + 1.0);
      status *= isEqual(values[2 * point + 1], p[1] + 2.0);
    }

    status *= std::filesystem::exists(dir / "obs-grid.pvd");
    status *= std::filesystem::exists(dir / "obs-grid_00000.vti");
    status *= std::filesystem::exists(dir / "obs-reference.xdmf");
    status *= std::filesystem::exists(dir / "obs-reference.h5");

    const TimeSeriesDataIn reference =
        TimeSeriesDataIn::read((dir / "obs-reference.xdmf").string());
    status *= (reference.numSteps() == 1);
    status *= isEqual(reference.time(0), 0.5);
    const auto& ref_velocity = reference.vectorField(0, "velocity");
    for (Index node = 0; node < reference.mesh().numNodes(); ++node)
    {
      status *= isEqual(ref_velocity[0][node],
                        reference.mesh().node(node)[0] + 1.0);
      status *= isEqual(ref_velocity[1][node],
                        reference.mesh().node(node)[1] + 2.0);
      status *= isEqual(ref_velocity[2][node], 0.0);
    }

    return status.report(__func__);
#endif
  }

  TestOutcome writesMultipleObservationCasesFromOneTrajectory()
  {
    TestStatus status;
    status = true;

#ifndef FEMX_HAS_HDF5
    return status.report(__func__);
#else
    const auto dir =
        std::filesystem::temp_directory_path() / "femx_make_obs_multi_tests";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir / "obs");

    Mesh mesh = Mesh::makeStructuredQuad(1, 1);
    Vector<Real> ux(mesh.numNodes());
    Vector<Real> uy(mesh.numNodes());
    Vector<Real> uz(mesh.numNodes());
    uz.setZero();

    TimeSeriesDataOut out;
    out.attachMesh(mesh);
    for (Index step = 0; step < 3; ++step)
    {
      out.beginStep(static_cast<Real>(step));
      for (Index node = 0; node < mesh.numNodes(); ++node)
      {
        ux[node] = mesh.node(node)[0] + static_cast<Real>(step);
        uy[node] = mesh.node(node)[1] + 2.0 * static_cast<Real>(step);
      }
      out.addNodalVectorField("velocity", ux, uy, uz);
    }
    out.write((dir / "trajectory").string());

    const auto config_path = dir / "Config.json";
    {
      std::ofstream config(config_path);
      config << R"({
  "input": {
    "trajectory": "trajectory.xdmf"
  },
  "output": {
    "file": "obs/common.txt",
    "write_reference": false
  },
  "obs1": {
    "type": "grid",
    "grid": {
      "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
      "counts": [2, 2, 1]
    },
    "components": [0]
  },
  "obs2": {
    "type": "grid",
    "grid": {
      "bounds": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
      "counts": [2, 1, 1]
    },
    "components": [1],
    "file": "obs/y.txt",
    "write_reference": false
  }
})";
    }

    const make_obs::Params prm =
        make_obs::loadConfig(config_path.string());
    status *= prm.observations.size() == 2;
    status *= prm.observations[0].name == "obs1";
    status *= prm.observations[1].name == "obs2";
    status *= prm.observations[0].output.file
              == (dir / "obs/common-obs1.txt").lexically_normal().string();
    status *= prm.observations[1].output.file
              == (dir / "obs/y.txt").lexically_normal().string();

    make_obs::writeTrajectoryObservationOutputs(prm);

    const auto obs1_path = dir / "obs/common-obs1.txt";
    const auto obs2_path = dir / "obs/y.txt";
    status *= std::filesystem::exists(obs1_path);
    status *= std::filesystem::exists(obs2_path);
    status *= !std::filesystem::exists(dir / "obs/common-obs1-reference.xdmf");
    status *= !std::filesystem::exists(dir / "obs/y-reference.xdmf");

    const inverse::TimeObservationData obs1 =
        inverse::readTimeObsData(obs1_path.string());
    const inverse::TimeObservationData obs2 =
        inverse::readTimeObsData(obs2_path.string());
    status *= obs1.numLevels() == 3;
    status *= obs1.numObservations() == 4;
    status *= obs2.numLevels() == 3;
    status *= obs2.numObservations() == 2;
    status *= isEqual(obs1.timeValue(0), 0.0);
    status *= isEqual(obs1.timeValue(2), 2.0);
    status *= isEqual(obs2[2][0], 4.0);

    return status.report(__func__);
#endif
  }

  TestOutcome rejectsNonGridObservationTypes()
  {
    TestStatus status;
    status = true;

    const auto dir =
        std::filesystem::temp_directory_path()
        / "femx_make_obs_config_reject_obs_type";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const auto rejects = [&](const std::string& type) {
      const auto config_path = dir / ("Config-" + type + ".json");
      {
        std::ofstream config(config_path);
        config << R"({
  "obs": {
    "type": ")" << type << R"("
  }
})";
      }

      try
      {
        (void) make_obs::loadConfig(config_path.string());
      }
      catch (const std::runtime_error& err)
      {
        return std::string(err.what()).find("obs.type")
               != std::string::npos;
      }
      return false;
    };

    status *= rejects("point");
    status *= rejects("cartesian");
    status *= rejects("cartesian_grid");

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running make-obs trajectory observation tests:\n";

  femx::tests::MakeObsTrajectoryObservationTests test;

  femx::tests::TestingResults result;
  result += test.samplesObservationFromTrajectory();
  result += test.writesMultipleObservationCasesFromOneTrajectory();
  result += test.rejectsNonGridObservationTypes();

  return result.summary();
}
