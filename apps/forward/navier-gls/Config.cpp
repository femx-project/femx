#include "Config.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
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

std::optional<Real> optionalReal(const nlohmann::json& node,
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
  return node.at(key).get<Real>();
}

std::array<Real, 3> parseVector3(const nlohmann::json& node,
                                 const std::string&    name)
{
  if (!node.is_array() || node.size() != 3)
  {
    throw std::runtime_error(name + " must be an array with 3 values");
  }

  std::array<Real, 3> values{};
  for (Index i = 0; i < 3; ++i)
  {
    values[i] = node.at(i).get<Real>();
  }
  return values;
}

Vector<Real> parseRealVector(const nlohmann::json& node,
                             const std::string&    name)
{
  if (!node.is_array())
  {
    throw std::runtime_error(name + " must be an array");
  }

  Vector<Real> values(static_cast<Index>(node.size()));
  for (Index i = 0; i < values.size(); ++i)
  {
    values[i] = node.at(i).get<Real>();
  }
  return values;
}

std::filesystem::path resolveConfigPath(const std::filesystem::path& config_dir,
                                        const std::string&           path)
{
  const std::filesystem::path candidate(path);
  if (candidate.is_absolute() || config_dir.empty())
  {
    return candidate;
  }
  return (config_dir / candidate).lexically_normal();
}

void parseVelocityTable(const std::filesystem::path& path,
                        VelocityParams&              velocity)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::runtime_error("Failed to open velocity table: " + path.string());
  }

  velocity.time.clear();
  velocity.value.clear();

  std::string line;
  Index       line_no = 0;
  while (std::getline(input, line))
  {
    ++line_no;
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#')
    {
      continue;
    }

    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    Real               t = 0.0;
    Real               v = 0.0;
    if (!(row >> t >> v))
    {
      throw std::runtime_error("Invalid velocity table row "
                               + std::to_string(line_no) + " in "
                               + path.string());
    }
    velocity.time.push_back(t);
    velocity.value.push_back(v);
  }
}

VelocityProfileParams parseVelocityProfile(const nlohmann::json& node)
{
  VelocityProfileParams profile;
  if (node.is_string())
  {
    profile.type = node.get<std::string>();
    return profile;
  }
  if (!node.is_object())
  {
    throw std::runtime_error(
        "Boundary velocity profile must be a string or object");
  }

  assign(node, "type", profile.type);
  assign(node, "radius", profile.radius);
  if (node.contains("center"))
  {
    if (node.at("center").is_string())
    {
      const auto center = node.at("center").get<std::string>();
      if (center != "auto")
      {
        throw std::runtime_error(
            "Boundary velocity profile center must be 'auto' or a vector");
      }
    }
    else
    {
      profile.center =
          parseVector3(node.at("center"), "Boundary velocity profile center");
    }
  }
  return profile;
}

VelocityParams parseVelocity(const nlohmann::json&        node,
                             const std::filesystem::path& config_dir,
                             const std::string&           name)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary " + name + " must be an object");
  }

  VelocityParams velocity;
  if (node.contains("table"))
  {
    const auto table_path =
        resolveConfigPath(config_dir, node.at("table").get<std::string>());
    parseVelocityTable(table_path, velocity);
  }

  if (node.contains("time"))
  {
    velocity.time = parseRealVector(node.at("time"), "Boundary velocity time");
  }
  else if (!node.contains("table"))
  {
    throw std::runtime_error("Boundary " + name + " requires time");
  }
  if (node.contains("value"))
  {
    velocity.value =
        parseRealVector(node.at("value"), "Boundary velocity value");
  }
  else if (!node.contains("table"))
  {
    throw std::runtime_error("Boundary " + name + " requires value");
  }

  assign(node, "area", velocity.area);
  assign(node, "period", velocity.period);
  assign(node, "quantity", velocity.quantity);
  if (node.contains("normal"))
  {
    velocity.normal =
        parseVector3(node.at("normal"), "Boundary velocity normal");
  }
  if (node.contains("profile"))
  {
    velocity.profile = parseVelocityProfile(node.at("profile"));
  }

  if (node.contains("interpolate"))
  {
    velocity.interp = node.at("interpolate").get<std::string>();
  }
  else if (node.contains("methodInterpolate"))
  {
    velocity.interp = node.at("methodInterpolate").get<std::string>();
  }
  else if (node.contains("methodinterpolate"))
  {
    velocity.interp = node.at("methodinterpolate").get<std::string>();
  }
  else if (node.contains("method"))
  {
    velocity.interp = node.at("method").get<std::string>();
  }

  return velocity;
}

BCsParams parseBoundaryCondition(const nlohmann::json&        node,
                                 const std::filesystem::path& config_dir)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary condition must be an object");
  }

  BCsParams cond;
  if (node.contains("physical"))
  {
    cond.tag = node.at("physical").get<Index>();
  }
  else if (node.contains("physical_tag"))
  {
    cond.tag = node.at("physical_tag").get<Index>();
  }
  else if (node.contains("tag"))
  {
    cond.tag = node.at("tag").get<Index>();
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
  if (node.contains("velocity") && node.contains("flowrate"))
  {
    throw std::runtime_error(
        "Boundary condition must not contain both velocity and flowrate");
  }
  if (node.contains("velocity"))
  {
    cond.velocity = parseVelocity(node.at("velocity"), config_dir, "velocity");
  }
  else if (node.contains("flowrate"))
  {
    cond.velocity =
        parseVelocity(node.at("flowrate"), config_dir, "flowrate");
    cond.velocity->quantity = "flowrate";
  }

  if (!cond.ux && !cond.uy && !cond.uz && !cond.p && !cond.velocity)
  {
    throw std::runtime_error(
        "Boundary condition needs at least one of ux, uy, uz, p, velocity, or flowrate");
  }
  return cond;
}

void validateVelocity(const VelocityParams& velocity)
{
  if (velocity.time.empty())
  {
    throw std::runtime_error("Boundary velocity time must not be empty");
  }
  if (velocity.time.size() != velocity.value.size())
  {
    throw std::runtime_error(
        "Boundary velocity time and value must have the same length");
  }
  if (velocity.area <= 0.0)
  {
    throw std::runtime_error("Boundary velocity area must be positive");
  }
  if (velocity.period < 0.0)
  {
    throw std::runtime_error("Boundary velocity period must be nonnegative");
  }
  for (Index i = 1; i < velocity.time.size(); ++i)
  {
    if (velocity.time[i] <= velocity.time[i - 1])
    {
      throw std::runtime_error(
          "Boundary velocity time values must be strictly increasing");
    }
  }

  Real normal_mag2 = 0.0;
  for (Real comp : velocity.normal)
  {
    normal_mag2 += comp * comp;
  }
  if (normal_mag2 <= 1.0e-28)
  {
    throw std::runtime_error("Boundary velocity normal must be nonzero");
  }

  if (velocity.interp != "constant" && velocity.interp != "nearest"
      && velocity.interp != "linear" && velocity.interp != "cubic")
  {
    throw std::runtime_error(
        "Boundary velocity interpolate must be 'constant', 'nearest', 'linear', or 'cubic'");
  }
  if (velocity.quantity != "flowrate"
      && velocity.quantity != "mean_velocity"
      && velocity.quantity != "max_velocity")
  {
    throw std::runtime_error(
        "Boundary velocity quantity must be 'flowrate', 'mean_velocity', or 'max_velocity'");
  }
  if (velocity.profile.type != "uniform"
      && velocity.profile.type != "poiseuille")
  {
    throw std::runtime_error(
        "Boundary velocity profile type must be 'uniform' or 'poiseuille'");
  }
  if (velocity.profile.type == "poiseuille" && velocity.profile.radius <= 0.0)
  {
    throw std::runtime_error(
        "Boundary velocity poiseuille profile requires a positive radius");
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
    if (cond.velocity)
    {
      validateVelocity(*cond.velocity);
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
  const auto root       = nlohmann::json::parse(input, nullptr, true, true);
  const auto config_dir = std::filesystem::path(path).parent_path();

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
  params.output.interval = std::max<Index>(1, params.output.interval);

  if (root.contains("bcs"))
  {
    const auto& boundaries = root.at("bcs");
    if (!boundaries.is_array())
    {
      throw std::runtime_error("Config bcs must be an array");
    }
    for (const auto& item : boundaries)
    {
      params.bcs.push_back(parseBoundaryCondition(item, config_dir));
    }
  }

  const std::filesystem::path mesh_path(params.mesh_file);
  if (!params.mesh_file.empty() && mesh_path.is_relative())
  {
    if (!config_dir.empty())
    {
      params.mesh_file = (config_dir / mesh_path).lexically_normal().string();
    }
  }

  validate(params);
  return params;
}

} // namespace femx
