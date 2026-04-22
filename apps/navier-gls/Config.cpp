#include "Config.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace refem
{

template <typename T>
void assign(const nlohmann::json& node,
            const char*           key,
            T&                    value)
{
  if (node.contains(key))
  {
    value = node.at(key).get<T>();
  }
}

std::optional<real_type> optionalReal(const nlohmann::json& node,
                                      const char*           key)
{
  if (!node.contains(key))
  {
    return std::nullopt;
  }
  if (!node.at(key).is_number())
  {
    throw std::runtime_error(std::string("Boundary value ") + key + " must be a number");
  }
  return node.at(key).get<real_type>();
}

std::array<real_type, 3> parseVector3(const nlohmann::json& node,
                                      const std::string&    name)
{
  if (!node.is_array() || node.size() != 3)
  {
    throw std::runtime_error(name + " must be an array with 3 values");
  }

  std::array<real_type, 3> values{};
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    values[i] = node.at(i).get<real_type>();
  }
  return values;
}

FlowRateParams parseFlowRate(const nlohmann::json& node)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary flowrate must be an object");
  }

  FlowRateParams flowrate;
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

  assign(node, "area", flowrate.area);
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

BCsParams parseBoundaryCondition(const nlohmann::json& node)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary condition must be an object");
  }

  BCsParams condition;
  if (node.contains("physical"))
  {
    condition.tag = node.at("physical").get<index_type>();
  }
  else if (node.contains("physical_tag"))
  {
    condition.tag = node.at("physical_tag").get<index_type>();
  }
  else if (node.contains("tag"))
  {
    condition.tag = node.at("tag").get<index_type>();
  }
  else
  {
    throw std::runtime_error(
        "Each boundary condition needs physical, physical_tag, or tag");
  }

  assign(node, "type", condition.type);
  condition.ux = optionalReal(node, "ux");
  condition.uy = optionalReal(node, "uy");
  condition.uz = optionalReal(node, "uz");
  condition.p  = optionalReal(node, "p");
  if (node.contains("flowrate"))
  {
    condition.flowrate = parseFlowRate(node.at("flowrate"));
  }

  if (!condition.ux && !condition.uy && !condition.uz && !condition.p
      && !condition.flowrate)
  {
    throw std::runtime_error(
        "Boundary condition needs at least one of ux, uy, uz, p, or flowrate");
  }
  return condition;
}

void validateFlowRate(const FlowRateParams& flowrate)
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

  if (flowrate.interpolate != "constant" && flowrate.interpolate != "nearest"
      && flowrate.interpolate != "linear" && flowrate.interpolate != "cubic")
  {
    throw std::runtime_error(
        "Boundary flowrate interpolate must be 'constant', 'nearest', 'linear', or 'cubic'");
  }
}

void validate(const Params& params)
{
  if (params.mesh_file.empty())
  {
    throw std::runtime_error("Config mesh.file is required");
  }
  if (params.time.steps <= 0 || params.time.dt <= 0.0)
  {
    throw std::runtime_error("Time steps and dt must be positive");
  }
  if (params.fluid.rho <= 0.0 || params.fluid.mu <= 0.0)
  {
    throw std::runtime_error("Fluid rho and mu must be positive");
  }
  if (params.solver.backend != "cpu" && params.solver.backend != "cuda")
  {
    throw std::runtime_error("Backend must be either 'cpu' or 'cuda'");
  }
  if (params.bcs.empty())
  {
    throw std::runtime_error("Config bcs must contain at least one boundary");
  }

  for (const auto& condition : params.bcs)
  {
    if (condition.tag <= 0)
    {
      throw std::runtime_error(
          "Boundary condition physical tag must be positive");
    }
    if (condition.type != "dirichlet")
    {
      throw std::runtime_error("Only dirichlet bcs are supported");
    }
    if (condition.flowrate)
    {
      validateFlowRate(*condition.flowrate);
    }
  }
}

CLOptions parseCommandLine(int argc, char* argv[])
{
  CLOptions options;

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
      std::cout << "Usage: navier-gls --config FILE\n";
      std::exit(0);
    }
    if (key == "--config")
    {
      options.config_file = requireValue(i, key);
      continue;
    }
    throw std::runtime_error("Unknown option: " + key);
  }
  if (options.config_file.empty())
  {
    throw std::runtime_error("Missing required option: --config FILE");
  }
  return options;
}

Params loadConfig(const std::string& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::runtime_error("Failed to open config file: " + path);
  }

  Params     params;
  const auto root = nlohmann::json::parse(input, nullptr, true, true);

  if (root.contains("mesh"))
  {
    assign(root.at("mesh"), "file", params.mesh_file);
  }
  if (root.contains("time"))
  {
    const auto& time = root.at("time");
    assign(time, "steps", params.time.steps);
    assign(time, "dt", params.time.dt);
  }
  if (root.contains("fluid"))
  {
    const auto& fluid = root.at("fluid");
    assign(fluid, "rho", params.fluid.rho);
    assign(fluid, "mu", params.fluid.mu);
  }
  if (root.contains("solver"))
  {
    assign(root.at("solver"), "backend", params.solver.backend);
  }
  if (root.contains("output"))
  {
    assign(root.at("output"), "interval", params.output.interval);
    assign(root.at("output"), "directory", params.output.directory);
  }
  params.output.interval = std::max<index_type>(1, params.output.interval);

  if (root.contains("bcs"))
  {
    const auto& boundaries = root.at("bcs");
    if (!boundaries.is_array())
    {
      throw std::runtime_error("Config bcs must be an array");
    }
    for (const auto& item : boundaries)
    {
      params.bcs.push_back(parseBoundaryCondition(item));
    }
  }

  const std::filesystem::path mesh_path(params.mesh_file);
  if (!params.mesh_file.empty() && mesh_path.is_relative())
  {
    const auto config_dir = std::filesystem::path(path).parent_path();
    if (!config_dir.empty())
    {
      params.mesh_file = (config_dir / mesh_path).lexically_normal().string();
    }
  }

  validate(params);
  return params;
}

} // namespace refem
