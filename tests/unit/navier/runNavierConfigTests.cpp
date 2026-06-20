#include <filesystem>
#include <fstream>
#include <iostream>

#include "BoundaryConditions.hpp"
#include "Config.hpp"
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/mesh/Mesh.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class NavierConfigTests : public TestBase
{
public:
  TestOutcome loadVelocityTableRelativeToConfig()
  {
    TestStatus status;
    status = true;

    const auto dir =
        std::filesystem::temp_directory_path() / "femx_navier_config_test";
    std::filesystem::create_directories(dir);

    {
      std::ofstream table(dir / "flow.csv");
      table << "# time,value\n";
      table << "0.0,1.0\n";
      table << "0.1,2.0\n";
      table << "0.3,1.0\n";
    }

    const auto config_path = dir / "Config.json";
    {
      std::ofstream config(config_path);
      config << R"({
  "mesh": {
    "file": "mesh.msh"
  },
  "output": {
    "interval": 3,
    "directory": "output"
  },
  "bcs": [
    {
      "physical": 4,
      "type": "dirichlet",
      "velocity": {
        "quantity": "max_velocity",
        "table": "flow.csv",
        "period": 0.3,
        "area": 2.0,
        "normal": [0.0, 1.0, 0.0],
        "profile": {
          "type": "poiseuille",
          "radius": 0.2,
          "center": [0.1, 0.2, 0.3]
        },
        "interpolate": "linear"
      }
    }
  ]
})";
    }

    const Params prm = loadConfig(config_path.string());
    if (prm.mesh_file != (dir / "mesh.msh").lexically_normal().string())
    {
      std::cout << "mesh path was not resolved relative to config\n";
      status = false;
    }
    status *= prm.output.interval == 3;
    status *= prm.output.directory
              == (dir / "output").lexically_normal().string();
    if (prm.bcs.size() != 1 || !prm.bcs[0].velocity)
    {
      std::cout << "velocity boundary was not loaded\n";
      status = false;
      return status.report(__func__);
    }

    const VelocityParams& velocity = *prm.bcs[0].velocity;
    if (velocity.time.size() != 3 || velocity.value.size() != 3)
    {
      std::cout << "velocity table size mismatch\n";
      status = false;
    }
    if (!isEqual(velocity.time[1], 0.1)
        || !isEqual(velocity.value[1], 2.0))
    {
      std::cout << "velocity table values were not parsed correctly\n";
      status = false;
    }
    if (!isEqual(velocity.period, 0.3) || !isEqual(velocity.area, 2.0))
    {
      std::cout << "velocity scalar settings were not parsed correctly\n";
      status = false;
    }
    if (!isEqual(velocity.normal[0], 0.0)
        || !isEqual(velocity.normal[1], 1.0)
        || !isEqual(velocity.normal[2], 0.0))
    {
      std::cout << "velocity normal was not parsed correctly\n";
      status = false;
    }
    if (velocity.quantity != "max_velocity"
        || velocity.profile.type != "poiseuille"
        || !isEqual(velocity.profile.radius, 0.2)
        || !velocity.profile.center
        || !isEqual((*velocity.profile.center)[2], 0.3))
    {
      std::cout << "velocity quantity/profile settings were not parsed correctly\n";
      status = false;
    }

    return status.report(__func__);
  }

  TestOutcome loadLegacyFlowrateAsVelocity()
  {
    TestStatus status;
    status = true;

    const auto dir =
        std::filesystem::temp_directory_path() / "femx_navier_config_test";
    std::filesystem::create_directories(dir);

    const auto config_path = dir / "LegacyConfig.json";
    {
      std::ofstream config(config_path);
      config << R"({
  "mesh": {
    "file": "mesh.msh"
  },
  "bcs": [
    {
      "physical": 4,
      "type": "dirichlet",
      "flowrate": {
        "time": [0.0, 0.1],
        "value": [1.0, 2.0],
        "area": 2.0,
        "normal": [1.0, 0.0, 0.0]
      }
    }
  ]
})";
    }

    const Params prm = loadConfig(config_path.string());
    if (prm.bcs.size() != 1 || !prm.bcs[0].velocity)
    {
      std::cout << "legacy flowrate boundary was not loaded\n";
      status = false;
      return status.report(__func__);
    }

    const VelocityParams& velocity = *prm.bcs[0].velocity;
    if (velocity.quantity != "flowrate" || velocity.profile.type != "uniform")
    {
      std::cout << "legacy flowrate was not mapped to uniform flowrate velocity\n";
      status = false;
    }

    return status.report(__func__);
  }

  TestOutcome poiseuilleProfileAppliesSpatialVelocity()
  {
    TestStatus status;
    status = true;

    Mesh mesh(3);
    mesh.addNode({0.0, 0.0, 0.0});
    mesh.addNode({0.0, 1.0, 0.0});
    mesh.addNode({0.0, 0.5, 0.0});
    mesh.addNode({1.0, 0.0, 0.0});
    mesh.addCell({0, 1, 2, 3},
                 Cell::Shape::Tetrahedron,
                 3,
                 1,
                 1,
                 "fluid");

    Mesh::BoundaryFacet inlet;
    inlet.dim          = 2;
    inlet.physical_tag = 4;
    inlet.shape        = Cell::Shape::Triangle;
    inlet.node_ids     = {0, 1, 2};
    mesh.addBoundaryFacet(inlet);

    LagrangeTetrahedronP1 elem;
    FESpace               u_space(&mesh, &elem, 3);
    FESpace               p_space(&mesh, &elem);
    u_space.setup();
    p_space.setup();

    MixedFESpace space;
    space.addField(u_space);
    space.addField(p_space);
    space.setup();

    VelocityParams velocity;
    velocity.time.resize(1);
    velocity.value.resize(1);
    velocity.time[0]        = 0.0;
    velocity.value[0]       = 2.0;
    velocity.quantity       = "max_velocity";
    velocity.normal         = {1.0, 0.0, 0.0};
    velocity.profile.type   = "poiseuille";
    velocity.profile.radius = 1.0;
    velocity.profile.center = std::array<Real, 3>{0.0, 0.0, 0.0};

    BCsParams inlet_bc;
    inlet_bc.tag      = 4;
    inlet_bc.velocity = velocity;

    const DirichletCondition bc =
        makeBoundaryCondition(space, {inlet_bc}, 0.0);

    const auto valueForDof = [&](Index dof, Real expected)
    {
      for (Index i = 0; i < bc.dofs().size(); ++i)
      {
        if (bc.dofs()[i] == dof)
        {
          if (!isEqual(bc.values()[i], expected))
          {
            std::cout << "Unexpected value for dof " << dof << ": expected "
                      << expected << ", got "
                      << bc.values()[i] << "\n";
            status = false;
          }
          return;
        }
      }
      std::cout << "Dof " << dof << " was not constrained\n";
      status = false;
    };

    const auto u_dof = space.field(0);
    valueForDof(u_dof.globalDof(0, 0), 2.0);
    valueForDof(u_dof.globalDof(1, 0), 0.0);
    valueForDof(u_dof.globalDof(2, 0), 1.5);
    valueForDof(u_dof.globalDof(0, 1), 0.0);
    valueForDof(u_dof.globalDof(0, 2), 0.0);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running Navier config tests:\n";

  femx::tests::NavierConfigTests test;

  femx::tests::TestingResults result;
  result += test.loadVelocityTableRelativeToConfig();
  result += test.loadLegacyFlowrateAsVelocity();
  result += test.poiseuilleProfileAppliesSpatialVelocity();

  return result.summary();
}
