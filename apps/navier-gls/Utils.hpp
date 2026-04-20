#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NavierGLS.hpp"
#include <nlohmann/json.hpp>
#include <refem/io/TimeSeriesDataOut.hpp>
#include <refem/linalg/Vector.hpp>
#include <refem/mesh/Mesh.hpp>
#include <refem/solver/ReSolverLinearSolver.hpp>

#ifndef REFEM_NAVIERSTOKES_CONFIG_FILE
#define REFEM_NAVIERSTOKES_CONFIG_FILE "Config.json"
#endif

using namespace refem;

struct Options
{
  std::string backend = "cpu";
  std::string config  = REFEM_NAVIERSTOKES_CONFIG_FILE;
  std::string mesh_file;
};

template <typename T>
void assignIfPresent(const nlohmann::json& node,
                     const char*           key,
                     T&                    value)
{
  if (node.contains(key))
  {
    value = node.at(key).get<T>();
  }
}

void loadConfigFile(const std::string& path,
                    Options&           options)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::runtime_error("Failed to open config file: " + path);
  }

  const auto config = nlohmann::json::parse(input, nullptr, true, true);

  if (config.contains("mesh"))
  {
    const auto& mesh = config.at("mesh");
    assignIfPresent(mesh, "file", options.mesh_file);
  }

  if (config.contains("time"))
  {
    const auto& time = config.at("time");
    assignIfPresent(time, "steps", steps);
    assignIfPresent(time, "dt", dt);
  }

  if (config.contains("physics"))
  {
    const auto& physics = config.at("physics");
    assignIfPresent(physics, "rho", rho);
    assignIfPresent(physics, "mu", mu);
    assignIfPresent(physics, "lid", lid);
    assignIfPresent(physics, "inlet_velocity", inlet_velocity);
  }

  if (config.contains("bcs"))
  {
    const auto& conditions = config.at("bcs");
    if (!conditions.is_array())
    {
      throw std::runtime_error("Config bcs must be an array");
    }

    bcs.clear();
    for (const auto& item : conditions)
    {
      BoundaryConditionSpec condition;
      if (item.contains("physical"))
      {
        condition.physical_tag = item.at("physical").get<index_type>();
      }
      else if (item.contains("physical_tag"))
      {
        condition.physical_tag = item.at("physical_tag").get<index_type>();
      }
      else if (item.contains("tag"))
      {
        condition.physical_tag = item.at("tag").get<index_type>();
      }
      else
      {
        throw std::runtime_error(
            "Each boundary condition needs physical, physical_tag, or tag");
      }

      assignIfPresent(item, "type", condition.type);
      if (item.contains("ux"))
      {
        condition.ux     = item.at("ux").get<real_type>();
        condition.has_ux = true;
      }
      if (item.contains("uy"))
      {
        condition.uy     = item.at("uy").get<real_type>();
        condition.has_uy = true;
      }
      if (item.contains("uz"))
      {
        condition.uz     = item.at("uz").get<real_type>();
        condition.has_uz = true;
      }
      if (item.contains("p"))
      {
        condition.p     = item.at("p").get<real_type>();
        condition.has_p = true;
      }

      bcs.push_back(condition);
    }
  }

  if (config.contains("output"))
  {
    assignIfPresent(config.at("output"), "interval", interval);
  }

  if (config.contains("solver"))
  {
    assignIfPresent(config.at("solver"), "backend", options.backend);
  }
}

void validateOptions(const Options& options)
{
  if (options.mesh_file.empty())
  {
    throw std::runtime_error("Config mesh.file is required");
  }

  if (steps <= 0 || dt <= 0.0 || rho <= 0.0 ||
      mu <= 0.0 || inlet_velocity < 0.0)
  {
    throw std::runtime_error("Invalid non-positive Navier-Stokes parameter");
  }

  for (const auto& condition : bcs)
  {
    if (condition.physical_tag <= 0)
    {
      throw std::runtime_error("Boundary condition physical tag must be positive");
    }
    if (condition.type != "dirichlet")
    {
      throw std::runtime_error(
          "Only dirichlet bcs are supported");
    }
    if (!condition.has_ux && !condition.has_uy &&
        !condition.has_uz && !condition.has_p)
    {
      throw std::runtime_error(
          "Boundary condition needs at least one of ux, uy, uz, or p");
    }
  }

  interval = std::max<index_type>(1, interval);

  if (options.backend != "cpu" && options.backend != "cuda")
  {
    throw std::runtime_error("Backend must be either 'cpu' or 'cuda'");
  }
}

Options parseOptions(int argc, char* argv[])
{
  Options options;

  const auto requireValue = [argc, argv](int& i, const std::string& key)
  {
    if (i + 1 >= argc)
    {
      throw std::runtime_error("Missing value for " + key);
    }
    return std::string(argv[++i]);
  };

  for (int i = 1; i < argc; ++i)
  {
    const std::string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      std::cout << "Usage: navier-gls [--config FILE]\n";
      std::exit(0);
    }
    else if (key == "--config")
    {
      options.config = requireValue(i, key);
    }
    else
    {
      throw std::runtime_error("Unknown option: " + key);
    }
  }

  loadConfigFile(options.config, options);
  if (!options.mesh_file.empty())
  {
    const std::filesystem::path mesh_path(options.mesh_file);
    if (mesh_path.is_relative())
    {
      const std::filesystem::path config_dir =
          std::filesystem::path(options.config).parent_path();
      if (!config_dir.empty())
      {
        options.mesh_file = (config_dir / mesh_path).lexically_normal().string();
      }
    }
  }
  validateOptions(options);
  return options;
}

bool isFinite(const Vector& x)
{
  for (index_type i = 0; i < x.size(); ++i)
  {
    if (!std::isfinite(x[i]))
    {
      return false;
    }
  }
  return true;
}

void setSolverOptions(ReSolveOptions& options)
{
  options.factor             = "none";
  options.refactor           = "none";
  options.ir                 = "none";
  options.max_iterations     = 5000;
  options.restart            = 200;
  options.relative_tolerance = 1.0e-10;
  options.solve              = "fgmres";
  options.precond            = "ilu0";
  options.flexible           = true;
}

void writeTimeSeriesOutput(const Mesh&                  mesh,
                           const std::vector<Snapshot>& snapshots)
{
  const std::string root = std::string(REFEM_NAVIERSTOKES_OUTPUT_DIR);

  TimeSeriesDataOut velocity_out;
  velocity_out.attachMesh(mesh);

  TimeSeriesDataOut pressure_out;
  pressure_out.attachMesh(mesh);

  for (const Snapshot& snapshot : snapshots)
  {
    velocity_out.beginStep(snapshot.time);
    velocity_out.addNodalVectorField("velocity",
                                     snapshot.ux,
                                     snapshot.uy,
                                     snapshot.uz);

    pressure_out.beginStep(snapshot.time);
    pressure_out.addNodalScalarField("pressure", snapshot.p);
  }

  velocity_out.write(root + "/velocity");
  pressure_out.write(root + "/pressure");
}
