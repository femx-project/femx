#include "Config.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace femx
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

  FlowRateParams flow;
  if (node.contains("time"))
  {
    flow.time = node.at("time").get<std::vector<real_type>>();
  }
  else
  {
    throw std::runtime_error("Boundary flowrate requires time");
  }
  if (node.contains("value"))
  {
    flow.value = node.at("value").get<std::vector<real_type>>();
  }
  else
  {
    throw std::runtime_error("Boundary flowrate requires value");
  }

  assign(node, "area", flow.area);
  if (node.contains("normal"))
  {
    flow.normal = parseVector3(node.at("normal"), "Boundary flowrate normal");
  }

  if (node.contains("interpolate"))
  {
    flow.interp = node.at("interpolate").get<std::string>();
  }
  else if (node.contains("methodInterpolate"))
  {
    flow.interp = node.at("methodInterpolate").get<std::string>();
  }
  else if (node.contains("methodinterpolate"))
  {
    flow.interp = node.at("methodinterpolate").get<std::string>();
  }
  else if (node.contains("method"))
  {
    flow.interp = node.at("method").get<std::string>();
  }

  return flow;
}

BCsParams parseBoundaryCondition(const nlohmann::json& node)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary condition must be an object");
  }

  BCsParams cond;
  if (node.contains("physical"))
  {
    cond.tag = node.at("physical").get<index_type>();
  }
  else if (node.contains("physical_tag"))
  {
    cond.tag = node.at("physical_tag").get<index_type>();
  }
  else if (node.contains("tag"))
  {
    cond.tag = node.at("tag").get<index_type>();
  }
  else
  {
    throw std::runtime_error(
        "Each boundary condition needs physical, physical_tag, or tag");
  }

  assign(node, "type", cond.type);
  cond.ux = optionalReal(node, "ux");
  cond.uy = optionalReal(node, "uy");
  cond.uz = optionalReal(node, "uz");
  cond.p  = optionalReal(node, "p");
  if (node.contains("flowrate"))
  {
    cond.flow = parseFlowRate(node.at("flowrate"));
  }

  if (!cond.ux && !cond.uy && !cond.uz && !cond.p && !cond.flow)
  {
    throw std::runtime_error(
        "Boundary condition needs at least one of ux, uy, uz, p, or flowrate");
  }
  return cond;
}

void validateFlowRate(const FlowRateParams& flow)
{
  if (flow.time.empty())
  {
    throw std::runtime_error("Boundary flowrate time must not be empty");
  }
  if (flow.time.size() != flow.value.size())
  {
    throw std::runtime_error(
        "Boundary flowrate time and value must have the same length");
  }
  if (flow.area <= 0.0)
  {
    throw std::runtime_error("Boundary flowrate area must be positive");
  }
  for (std::size_t i = 1; i < flow.time.size(); ++i)
  {
    if (flow.time[i] <= flow.time[i - 1])
    {
      throw std::runtime_error(
          "Boundary flowrate time values must be strictly increasing");
    }
  }

  real_type normal_mag2 = 0.0;
  for (real_type comp : flow.normal)
  {
    normal_mag2 += comp * comp;
  }
  if (normal_mag2 <= 1.0e-28)
  {
    throw std::runtime_error("Boundary flowrate normal must be nonzero");
  }

  if (flow.interp != "constant" && flow.interp != "nearest"
      && flow.interp != "linear" && flow.interp != "cubic")
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

  for (const auto& cond : params.bcs)
  {
    if (cond.tag <= 0)
    {
      throw std::runtime_error(
          "Boundary condition physical tag must be positive");
    }
    if (cond.type != "dirichlet")
    {
      throw std::runtime_error("Only dirichlet bcs are supported");
    }
    if (cond.flow)
    {
      validateFlowRate(*cond.flow);
    }
  }
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

} // namespace femx
