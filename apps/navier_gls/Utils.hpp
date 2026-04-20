#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

struct Options
{
  std::string backend = "cpu";
  std::string config  = REFEM_NAVIERSTOKES_CONFIG_FILE;
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
    assignIfPresent(mesh, "nx", refem::nx);
    assignIfPresent(mesh, "ny", refem::ny);
  }

  if (config.contains("time"))
  {
    const auto& time = config.at("time");
    assignIfPresent(time, "steps", refem::steps);
    assignIfPresent(time, "dt", refem::dt);
  }

  if (config.contains("physics"))
  {
    const auto& physics = config.at("physics");
    assignIfPresent(physics, "rho", refem::rho);
    assignIfPresent(physics, "mu", refem::mu);
    assignIfPresent(physics, "lid", refem::lid);
  }

  if (config.contains("stability"))
  {
    assignIfPresent(config.at("stability"), "max_cfl", refem::max_cfl);
  }

  if (config.contains("output"))
  {
    assignIfPresent(config.at("output"), "interval", refem::interval);
  }

  if (config.contains("solver"))
  {
    assignIfPresent(config.at("solver"), "backend", options.backend);
  }
}

void validateOptions(const Options& options)
{
  if (refem::nx <= 0 || refem::ny <= 0 || refem::steps <= 0 ||
      refem::dt <= 0.0 || refem::max_cfl <= 0.0 || refem::rho <= 0.0 ||
      refem::mu <= 0.0)
  {
    throw std::runtime_error("Invalid non-positive Navier-Stokes parameter");
  }

  refem::interval = std::max<refem::index_type>(1, refem::interval);

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
      std::cout << "Usage: navier_gls [--config FILE]\n";
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
  validateOptions(options);
  return options;
}

bool isFinite(const refem::Vector& x)
{
  for (refem::index_type i = 0; i < x.size(); ++i)
  {
    if (!std::isfinite(x[i]))
    {
      return false;
    }
  }
  return true;
}

void setSolverOptions(refem::ReSolveOptions& options)
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

void writeTimeSeriesOutput(const refem::Mesh&                  mesh,
                           const std::vector<refem::Snapshot>& snapshots)
{
  const std::string root = std::string(REFEM_NAVIERSTOKES_OUTPUT_DIR);

  refem::TimeSeriesDataOut velocity_out;
  velocity_out.attachMesh(mesh);

  refem::TimeSeriesDataOut pressure_out;
  pressure_out.attachMesh(mesh);

  for (const refem::Snapshot& snapshot : snapshots)
  {
    velocity_out.beginStep(snapshot.time);
    velocity_out.addNodalVectorField("velocity", snapshot.ux, snapshot.uy);

    pressure_out.beginStep(snapshot.time);
    pressure_out.addNodalScalarField("pressure", snapshot.p);
  }

  velocity_out.write(root + "/velocity");
  pressure_out.write(root + "/pressure");
}
