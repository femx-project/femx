#include "ForwardConfig.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
using nlohmann::json;

namespace femx::model::ns
{

void checkKeys(const json&                        node,
               std::initializer_list<const char*> keys,
               const char*                        name)
{
  for (const auto& item : node.items())
  {
    bool known = false;
    for (const char* key : keys)
    {
      if (item.key() == key)
      {
        known = true;
        break;
      }
    }
    if (!known)
    {
      throw std::runtime_error(
          std::string(name) + " contains unknown option '" + item.key() + "'");
    }
  }
}

template <typename T>
void assign(const json& node,
            const char* key,
            T&          value)
{
  if (node.contains(key))
  {
    value = node.at(key).get<T>();
  }
}

std::optional<Real> optionalReal(const json& node,
                                 const char* key)
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

std::array<Real, 3> parseVector3(const json&        node,
                                 const std::string& name)
{
  if (!node.is_array() || node.size() != 3)
  {
    throw std::runtime_error(name + " must be an array with 3 values");
  }

  std::array<Real, 3> vals{};
  for (Index i = 0; i < 3; ++i)
  {
    vals[i] = node.at(i).get<Real>();
  }
  return vals;
}

HostVector parseRealVector(const json&        node,
                           const std::string& name)
{
  if (!node.is_array())
  {
    throw std::runtime_error(name + " must be an array");
  }

  HostVector vals(static_cast<Index>(node.size()));
  for (Index i = 0; i < vals.size(); ++i)
  {
    vals[i] = node.at(i).get<Real>();
  }
  return vals;
}

std::filesystem::path resolveConfigPath(const std::filesystem::path& cfg_dir,
                                        const std::string&           path)
{
  const std::filesystem::path candidate(path);
  if (candidate.is_absolute() || cfg_dir.empty())
  {
    return candidate;
  }
  return (cfg_dir / candidate).lexically_normal();
}

Index stepsForEndTime(Real end_time,
                      Real dt)
{
  if (!std::isfinite(end_time) || end_time <= 0.0)
  {
    throw std::runtime_error("Config time.end_time must be positive");
  }
  if (!std::isfinite(dt) || dt <= 0.0)
  {
    throw std::runtime_error("Config time.dt must be positive");
  }

  const Real scaled = end_time / dt;
  if (!std::isfinite(scaled) || scaled <= 0.0
      || scaled > static_cast<Real>(std::numeric_limits<Index>::max()))
  {
    throw std::runtime_error("Config time.end_time / dt is out of range");
  }

  const Real eps = 64.0 * std::numeric_limits<Real>::epsilon()
                   * (std::abs(scaled) + 1.0);
  return static_cast<Index>(ceil(scaled - eps));
}

void parseConvergenceConfig(const json&        node,
                            ConvergenceParams& convergence)
{
  if (node.is_boolean())
  {
    convergence.enabled = node.get<bool>();
    return;
  }
  if (!node.is_object())
  {
    throw std::runtime_error("Config time.convergence must be a boolean or object");
  }
  checkKeys(node,
            {"enabled", "vel_rel_tol", "min_steps"},
            "Config time.convergence");

  convergence.enabled = true;
  assign(node, "enabled", convergence.enabled);
  assign(node, "vel_rel_tol", convergence.vel_rel_tol);
  assign(node, "min_steps", convergence.min_steps);
}

void parseTimeConfig(const json& node,
                     TimeParams& time)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config time must be an object");
  }
  checkKeys(node,
            {"dt", "steps", "end_time", "convergence"},
            "Config time");

  assign(node, "dt", time.dt);
  assign(node, "steps", time.steps);
  if (node.contains("end_time"))
  {
    const Index derived_steps =
        stepsForEndTime(node.at("end_time").get<Real>(), time.dt);
    if (node.contains("steps") && time.steps != derived_steps)
    {
      throw std::runtime_error(
          "Config time.steps and time.end_time disagree for the configured dt");
    }
    time.steps = derived_steps;
  }
  if (node.contains("convergence"))
  {
    parseConvergenceConfig(node.at("convergence"), time.convergence);
  }
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
  while (getline(input, line))
  {
    ++line_no;
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#')
    {
      continue;
    }

    replace(line.begin(), line.end(), ',', ' ');
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

VelocityProfileParams parseVelocityProfile(const json& node)
{
  VelocityProfileParams prof;
  if (node.is_string())
  {
    prof.type = node.get<std::string>();
    return prof;
  }
  if (!node.is_object())
  {
    throw std::runtime_error(
        "Boundary velocity profile must be a string or object");
  }

  assign(node, "type", prof.type);
  assign(node, "radius", prof.rad);
  if (node.contains("center"))
  {
    if (node.at("center").is_string())
    {
      const auto cen = node.at("center").get<std::string>();
      if (cen != "auto")
      {
        throw std::runtime_error(
            "Boundary velocity profile center must be 'auto' or a vector");
      }
    }
    else
    {
      prof.cen = parseVector3(node.at("center"), "Boundary velocity profile center");
    }
  }
  return prof;
}

void assignVelocityInterpolation(const json&     node,
                                 VelocityParams& velocity)
{
  assign(node, "interpolate", velocity.interp);
}

Real parseVelocityScalar(const json& node)
{
  if (node.contains("value"))
  {
    return node.at("value").get<Real>();
  }
  throw std::runtime_error(
      "Boundary velocity time profile requires value");
}

void parseVelocityTimeProfile(const json&                  node,
                              const std::filesystem::path& cfg_dir,
                              VelocityParams&              velocity)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary velocity time profile must be an object");
  }
  checkKeys(node,
            {"quantity",
             "type",
             "table",
             "period",
             "value",
             "amplitude",
             "interpolate"},
            "Boundary velocity time profile");

  assign(node, "quantity", velocity.qty);

  std::string type = node.contains("table") ? "table" : "uniform";
  assign(node, "type", type);
  if (type == "table")
  {
    if (!node.contains("table"))
    {
      throw std::runtime_error(
          "Boundary velocity table time profile requires table");
    }
    const auto table_path =
        resolveConfigPath(cfg_dir, node.at("table").get<std::string>());
    parseVelocityTable(table_path, velocity);
    assign(node, "period", velocity.per);
  }
  else if (type == "uniform")
  {
    velocity.time  = {0.0};
    velocity.value = {parseVelocityScalar(node)};
    assign(node, "period", velocity.per);
  }
  else if (type == "sine")
  {
    Real per = velocity.per > 0.0 ? velocity.per : 1.0;
    assign(node, "period", per);
    if (!std::isfinite(per) || per <= 0.0)
    {
      throw std::runtime_error(
          "Boundary velocity sine time profile period must be positive");
    }

    Real amplitude = 0.35;
    assign(node, "amplitude", amplitude);

    const Real base = parseVelocityScalar(node);
    velocity.per    = per;
    velocity.time =
        {0.0, 0.25 * per, 0.5 * per, 0.75 * per, per};
    velocity.value =
        {base,
         base * (1.0 + amplitude),
         base,
         base * (1.0 - amplitude),
         base};
    velocity.interp = "cubic";
  }
  else
  {
    throw std::runtime_error(
        "Boundary velocity time.type must be 'sine', 'uniform', or 'table'");
  }

  assignVelocityInterpolation(node, velocity);
}

void parseVelocitySpaceProfile(const json&     node,
                               VelocityParams& velocity)
{
  velocity.prof = parseVelocityProfile(node);
  if (!node.is_object())
  {
    return;
  }
  checkKeys(node,
            {"type", "radius", "center", "area", "quantity", "normal"},
            "Boundary velocity space profile");

  assign(node, "area", velocity.area);
  assign(node, "quantity", velocity.qty);
  if (node.contains("normal"))
  {
    velocity.nrm =
        parseVector3(node.at("normal"), "Boundary velocity space normal");
  }
}

VelocityParams parseVelocity(const json&                  node,
                             const std::filesystem::path& cfg_dir)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary velocity must be an object");
  }
  checkKeys(node,
            {"area",
             "period",
             "quantity",
             "normal",
             "profile",
             "table",
             "time",
             "value",
             "space",
             "interpolate"},
            "Boundary velocity");

  VelocityParams velocity;
  assign(node, "area", velocity.area);
  assign(node, "period", velocity.per);
  assign(node, "quantity", velocity.qty);
  if (node.contains("normal"))
  {
    velocity.nrm =
        parseVector3(node.at("normal"), "Boundary velocity normal");
  }
  if (node.contains("profile"))
  {
    velocity.prof = parseVelocityProfile(node.at("profile"));
  }

  if (node.contains("table"))
  {
    const auto table_path =
        resolveConfigPath(cfg_dir, node.at("table").get<std::string>());
    parseVelocityTable(table_path, velocity);
  }

  if (node.contains("time"))
  {
    const auto& time = node.at("time");
    if (time.is_object())
    {
      parseVelocityTimeProfile(time, cfg_dir, velocity);
    }
    else
    {
      velocity.time = parseRealVector(time, "Boundary velocity time");
    }
  }
  else if (!node.contains("table"))
  {
    throw std::runtime_error("Boundary velocity requires time");
  }
  if (node.contains("value"))
  {
    velocity.value =
        parseRealVector(node.at("value"), "Boundary velocity value");
  }
  else if (!node.contains("table")
           && !(node.contains("time") && node.at("time").is_object()))
  {
    throw std::runtime_error("Boundary velocity requires value");
  }

  if (node.contains("space"))
  {
    parseVelocitySpaceProfile(node.at("space"), velocity);
  }
  assignVelocityInterpolation(node, velocity);

  return velocity;
}

BCsParams parseDirichletBC(const json&                  node,
                           const std::filesystem::path& cfg_dir)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary condition must be an object");
  }
  checkKeys(node,
            {"name", "physical", "type", "ux", "uy", "uz", "p", "velocity"},
            "Boundary condition");

  BCsParams cond;
  if (!node.contains("physical"))
  {
    throw std::runtime_error(
        "Each boundary condition needs physical");
  }
  cond.tag = node.at("physical").get<Index>();

  assign(node, "type", cond.type);
  cond.ux = optionalReal(node, "ux");
  cond.uy = optionalReal(node, "uy");
  cond.uz = optionalReal(node, "uz");
  cond.p  = optionalReal(node, "p");
  if (node.contains("velocity"))
  {
    cond.velocity = parseVelocity(node.at("velocity"), cfg_dir);
  }

  if (!cond.ux && !cond.uy && !cond.uz && !cond.p && !cond.velocity)
  {
    throw std::runtime_error(
        "Boundary condition needs at least one of ux, uy, uz, p, or velocity");
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
  if (velocity.per < 0.0)
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
  for (Real comp : velocity.nrm)
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
  if (velocity.qty != "flowrate"
      && velocity.qty != "mean_velocity"
      && velocity.qty != "max_velocity")
  {
    throw std::runtime_error(
        "Boundary velocity quantity must be 'flowrate', 'mean_velocity', or 'max_velocity'");
  }
  if (velocity.prof.type != "uniform"
      && velocity.prof.type != "poiseuille")
  {
    throw std::runtime_error(
        "Boundary velocity profile type must be 'uniform' or 'poiseuille'");
  }
  if (velocity.prof.type == "poiseuille" && velocity.prof.rad <= 0.0)
  {
    throw std::runtime_error(
        "Boundary velocity poiseuille profile requires a positive radius");
  }
}

void validate(const Params& prm)
{
  if (prm.mesh_file.empty())
  {
    throw std::runtime_error("Config mesh.file is required");
  }
  if (prm.time.steps <= 0 || prm.time.dt <= 0.0)
  {
    throw std::runtime_error("Time steps and dt must be positive");
  }
  if (prm.time.convergence.enabled)
  {
    if (!std::isfinite(prm.time.convergence.vel_rel_tol)
        || prm.time.convergence.vel_rel_tol <= 0.0)
    {
      throw std::runtime_error(
          "Config time.convergence velocity relative tolerance must be positive");
    }
    if (prm.time.convergence.min_steps < 1)
    {
      throw std::runtime_error("Config time.convergence min_steps must be positive");
    }
  }
  if (prm.fluid.rho <= 0.0 || prm.fluid.mu <= 0.0)
  {
    throw std::runtime_error("Fluid rho and mu must be positive");
  }
  if (prm.solver.method != "direct"
      && prm.solver.method != "iterative")
  {
    throw std::runtime_error("Solver method must be either 'direct' or 'iterative'");
  }
  if (prm.solver.solve != "fgmres"
      && prm.solver.solve != "randgmres")
  {
    throw std::runtime_error("Solver solve must be either 'fgmres' or 'randgmres'");
  }
  if (prm.solver.preconditioner != "ilu0"
      && prm.solver.preconditioner != "none")
  {
    throw std::runtime_error(
        "Solver preconditioner must be either 'ilu0' or 'none'");
  }
  if (prm.solver.gram_schmidt != "cgs1"
      && prm.solver.gram_schmidt != "cgs2"
      && prm.solver.gram_schmidt != "mgs"
      && prm.solver.gram_schmidt != "mgs_two_sync"
      && prm.solver.gram_schmidt != "mgs_pm")
  {
    throw std::runtime_error(
        "Solver gram_schmidt must be 'cgs1', 'cgs2', 'mgs', 'mgs_two_sync', or 'mgs_pm'");
  }
  if (prm.solver.sketching != "count" && prm.solver.sketching != "fwht")
  {
    throw std::runtime_error("Solver sketching must be either 'count' or 'fwht'");
  }
  if (prm.solver.max_itrs <= 0 || prm.solver.restart <= 0)
  {
    throw std::runtime_error(
        "Solver max_itrs and restart must be positive");
  }
  if (!std::isfinite(prm.solver.relative_tolerance)
      || prm.solver.relative_tolerance <= 0.0)
  {
    throw std::runtime_error("Solver relative_tolerance must be positive");
  }
  if (prm.bcs.empty())
  {
    throw std::runtime_error("Config bcs must contain at least one boundary");
  }

  for (const auto& cond : prm.bcs)
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

  Params     prm;
  const auto root    = json::parse(input, nullptr, true, true);
  const auto cfg_dir = std::filesystem::path(path).parent_path();
  if (!root.is_object())
  {
    throw std::runtime_error("Config root must be an object");
  }
  checkKeys(root,
            {"mesh", "time", "fluid", "solver", "output", "bcs"},
            "Config root");

  if (root.contains("mesh"))
  {
    const auto& mesh = root.at("mesh");
    if (!mesh.is_object())
    {
      throw std::runtime_error("Config mesh must be an object");
    }
    checkKeys(mesh, {"file"}, "Config mesh");
    assign(mesh, "file", prm.mesh_file);
  }
  if (root.contains("time"))
  {
    parseTimeConfig(root.at("time"), prm.time);
  }
  if (root.contains("fluid"))
  {
    const auto& fluid = root.at("fluid");
    if (!fluid.is_object())
    {
      throw std::runtime_error("Config fluid must be an object");
    }
    checkKeys(fluid, {"rho", "mu"}, "Config fluid");
    assign(fluid, "rho", prm.fluid.rho);
    assign(fluid, "mu", prm.fluid.mu);
  }
  if (root.contains("solver"))
  {
    const auto& solver = root.at("solver");
    if (!solver.is_object())
    {
      throw std::runtime_error("Config solver must be an object");
    }
    checkKeys(solver,
              {"method",
               "solve",
               "preconditioner",
               "gram_schmidt",
               "sketching",
               "restart",
               "max_itrs",
               "relative_tolerance",
               "flexible"},
              "Config solver");
    assign(solver, "method", prm.solver.method);
    assign(solver, "solve", prm.solver.solve);
    assign(solver, "preconditioner", prm.solver.preconditioner);
    assign(solver, "gram_schmidt", prm.solver.gram_schmidt);
    assign(solver, "sketching", prm.solver.sketching);
    assign(solver, "restart", prm.solver.restart);
    assign(solver, "max_itrs", prm.solver.max_itrs);
    assign(solver, "relative_tolerance", prm.solver.relative_tolerance);
    assign(solver, "flexible", prm.solver.flexible);
  }
  if (root.contains("output"))
  {
    const auto& output = root.at("output");
    if (!output.is_object())
    {
      throw std::runtime_error("Config output must be an object");
    }
    checkKeys(output,
              {"enabled", "interval", "directory"},
              "Config output");
    assign(output, "enabled", prm.output.enabled);
    assign(output, "interval", prm.output.interval);
    const bool has_directory = output.contains("directory");
    assign(output, "directory", prm.output.directory);
    if (has_directory && !prm.output.directory.empty())
    {
      prm.output.directory =
          resolveConfigPath(cfg_dir, prm.output.directory).string();
    }
  }
  prm.output.interval = std::max<Index>(1, prm.output.interval);

  if (root.contains("bcs"))
  {
    const auto& boundaries = root.at("bcs");
    if (!boundaries.is_array())
    {
      throw std::runtime_error("Config bcs must be an array");
    }
    for (const auto& item : boundaries)
    {
      prm.bcs.push_back(parseDirichletBC(item, cfg_dir));
    }
  }

  const std::filesystem::path mesh_path(prm.mesh_file);
  if (!prm.mesh_file.empty() && mesh_path.is_relative())
  {
    if (!cfg_dir.empty())
    {
      prm.mesh_file = (cfg_dir / mesh_path).lexically_normal().string();
    }
  }

  validate(prm);
  return prm;
}

} // namespace femx::model::ns
