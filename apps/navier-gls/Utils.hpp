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

#include <nlohmann/json.hpp>

#include "NavierGLS.hpp"
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

index_type parseAxis(const nlohmann::json& node)
{
  if (node.is_number_integer())
  {
    return node.get<index_type>();
  }
  if (!node.is_string())
  {
    throw std::runtime_error("Boundary profile axis must be 0, 1, 2, x, y, or z");
  }

  const std::string axis = node.get<std::string>();
  if (axis == "x")
  {
    return 0;
  }
  if (axis == "y")
  {
    return 1;
  }
  if (axis == "z")
  {
    return 2;
  }
  throw std::runtime_error("Boundary profile axis must be 0, 1, 2, x, y, or z");
}

BoundaryConditionSpec::TimeProfile parseTimeProfile(const nlohmann::json& node)
{
  BoundaryConditionSpec::TimeProfile time;
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary time profile must be an object");
  }

  assignIfPresent(node, "profile", time.profile);
  assignIfPresent(node, "value", time.value);
  assignIfPresent(node, "from", time.from);
  assignIfPresent(node, "to", time.to);
  assignIfPresent(node, "t0", time.t0);
  assignIfPresent(node, "t1", time.t1);
  assignIfPresent(node, "mean", time.mean);
  assignIfPresent(node, "amplitude", time.amplitude);
  assignIfPresent(node, "frequency", time.frequency);
  assignIfPresent(node, "phase", time.phase);
  return time;
}

BoundaryConditionSpec::Value parseBoundaryValue(const nlohmann::json& node)
{
  BoundaryConditionSpec::Value value;
  if (node.is_number())
  {
    value.value = node.get<real_type>();
    return value;
  }
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary value must be a number or an object");
  }

  assignIfPresent(node, "profile", value.profile);
  if (node.contains("value"))
  {
    value.value = node.at("value").get<real_type>();
  }
  else if (node.contains("max"))
  {
    value.value = node.at("max").get<real_type>();
  }

  if (node.contains("axis"))
  {
    value.axis = parseAxis(node.at("axis"));
  }
  if (node.contains("time"))
  {
    value.time = parseTimeProfile(node.at("time"));
  }
  return value;
}

std::array<real_type, dim> parseVector3(const nlohmann::json& node,
                                        const std::string&    name)
{
  if (!node.is_array() || node.size() != static_cast<std::size_t>(dim))
  {
    throw std::runtime_error(name + " must be an array with 3 values");
  }

  std::array<real_type, dim> values{};
  for (index_type i = 0; i < dim; ++i)
  {
    values[static_cast<std::size_t>(i)] = node.at(i).get<real_type>();
  }
  return values;
}

BoundaryConditionSpec::FlowRate parseFlowRate(const nlohmann::json& node)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary flowrate must be an object");
  }

  BoundaryConditionSpec::FlowRate flowrate;
  if (node.contains("time"))
  {
    flowrate.time = node.at("time").get<std::vector<real_type>>();
  }
  else
  {
    throw std::runtime_error("Boundary flowrate requires time");
  }
  if (node.contains("value"))
  {
    flowrate.value = node.at("value").get<std::vector<real_type>>();
  }
  else
  {
    throw std::runtime_error("Boundary flowrate requires value");
  }

  assignIfPresent(node, "area", flowrate.area);
  if (node.contains("normal"))
  {
    flowrate.normal = parseVector3(node.at("normal"), "Boundary flowrate normal");
  }

  if (node.contains("interpolate"))
  {
    flowrate.interpolate = node.at("interpolate").get<std::string>();
  }
  else if (node.contains("methodInterpolate"))
  {
    flowrate.interpolate = node.at("methodInterpolate").get<std::string>();
  }
  else if (node.contains("methodinterpolate"))
  {
    flowrate.interpolate = node.at("methodinterpolate").get<std::string>();
  }
  else if (node.contains("method"))
  {
    flowrate.interpolate = node.at("method").get<std::string>();
  }

  return flowrate;
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
        condition.ux     = parseBoundaryValue(item.at("ux"));
        condition.has_ux = true;
      }
      if (item.contains("uy"))
      {
        condition.uy     = parseBoundaryValue(item.at("uy"));
        condition.has_uy = true;
      }
      if (item.contains("uz"))
      {
        condition.uz     = parseBoundaryValue(item.at("uz"));
        condition.has_uz = true;
      }
      if (item.contains("p"))
      {
        condition.p     = parseBoundaryValue(item.at("p"));
        condition.has_p = true;
      }
      if (item.contains("flowrate"))
      {
        condition.flowrate     = parseFlowRate(item.at("flowrate"));
        condition.has_flowrate = true;
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

void validateFlowRate(const BoundaryConditionSpec::FlowRate& flowrate)
{
  if (flowrate.time.empty())
  {
    throw std::runtime_error("Boundary flowrate time must not be empty");
  }
  if (flowrate.time.size() != flowrate.value.size())
  {
    throw std::runtime_error(
        "Boundary flowrate time and value must have the same length");
  }
  if (flowrate.area <= 0.0)
  {
    throw std::runtime_error("Boundary flowrate area must be positive");
  }
  for (std::size_t i = 1; i < flowrate.time.size(); ++i)
  {
    if (flowrate.time[i] <= flowrate.time[i - 1])
    {
      throw std::runtime_error(
          "Boundary flowrate time values must be strictly increasing");
    }
  }

  real_type normal_mag2 = 0.0;
  for (real_type component : flowrate.normal)
  {
    normal_mag2 += component * component;
  }
  if (normal_mag2 <= 1.0e-28)
  {
    throw std::runtime_error("Boundary flowrate normal must be nonzero");
  }

  if (flowrate.interpolate != "constant" && flowrate.interpolate != "nearest" && flowrate.interpolate != "linear" && flowrate.interpolate != "cubic")
  {
    throw std::runtime_error(
        "Boundary flowrate interpolate must be 'constant', 'nearest', 'linear', or 'cubic'");
  }
}

void validateBoundaryValue(const BoundaryConditionSpec::Value& value)
{
  if (value.profile != "constant" && value.profile != "parabolic")
  {
    throw std::runtime_error(
        "Boundary value profile must be 'constant' or 'parabolic'");
  }
  if (value.axis < 0 || value.axis >= 3)
  {
    throw std::runtime_error("Boundary value axis must be 0, 1, or 2");
  }

  const auto& time = value.time;
  if (time.profile != "constant" && time.profile != "ramp" && time.profile != "sin")
  {
    throw std::runtime_error(
        "Boundary time profile must be 'constant', 'ramp', or 'sin'");
  }
  if (time.profile == "ramp" && time.t1 < time.t0)
  {
    throw std::runtime_error("Boundary ramp time profile requires t1 >= t0");
  }
}

void validateOptions(const Options& options)
{
  if (options.mesh_file.empty())
  {
    throw std::runtime_error("Config mesh.file is required");
  }

  if (steps <= 0 || dt <= 0.0 || rho <= 0.0 || mu <= 0.0 || inlet_velocity < 0.0)
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
    if (!condition.has_ux && !condition.has_uy && !condition.has_uz && !condition.has_p && !condition.has_flowrate)
    {
      throw std::runtime_error(
          "Boundary condition needs at least one of ux, uy, uz, p, or flowrate");
    }
    if (condition.has_ux)
    {
      validateBoundaryValue(condition.ux);
    }
    if (condition.has_uy)
    {
      validateBoundaryValue(condition.uy);
    }
    if (condition.has_uz)
    {
      validateBoundaryValue(condition.uz);
    }
    if (condition.has_p)
    {
      validateBoundaryValue(condition.p);
    }
    if (condition.has_flowrate)
    {
      validateFlowRate(condition.flowrate);
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
