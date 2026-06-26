#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using namespace std;
using nlohmann::json;

namespace femx
{

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

optional<Real> optionalReal(const json& node,
                            const char* key)
{
  if (!node.contains(key))
  {
    return nullopt;
  }
  if (!node.at(key).is_number())
  {
    throw runtime_error(string("Boundary value ") + key + " must be a number");
  }
  return node.at(key).get<Real>();
}

array<Real, 3> parseVector3(const json&   node,
                            const string& name)
{
  if (!node.is_array() || node.size() != 3)
  {
    throw runtime_error(name + " must be an array with 3 values");
  }

  array<Real, 3> vals{};
  for (Index i = 0; i < 3; ++i)
  {
    vals[i] = node.at(i).get<Real>();
  }
  return vals;
}

Vector<Real> parseRealVector(const json&   node,
                             const string& name)
{
  if (!node.is_array())
  {
    throw runtime_error(name + " must be an array");
  }

  Vector<Real> vals(static_cast<Index>(node.size()));
  for (Index i = 0; i < vals.size(); ++i)
  {
    vals[i] = node.at(i).get<Real>();
  }
  return vals;
}

filesystem::path resolveConfigPath(const filesystem::path& cfg_dir,
                                   const string&           path)
{
  const filesystem::path candidate(path);
  if (candidate.is_absolute() || cfg_dir.empty())
  {
    return candidate;
  }
  return (cfg_dir / candidate).lexically_normal();
}

Index stepsForEndTime(Real end_time,
                      Real dt)
{
  if (!isfinite(end_time) || end_time <= 0.0)
  {
    throw runtime_error("Config time.end_time must be positive");
  }
  if (!isfinite(dt) || dt <= 0.0)
  {
    throw runtime_error("Config time.dt must be positive");
  }

  const Real scaled = end_time / dt;
  if (!isfinite(scaled) || scaled <= 0.0
      || scaled > static_cast<Real>(numeric_limits<Index>::max()))
  {
    throw runtime_error("Config time.end_time / dt is out of range");
  }

  const Real eps = 64.0 * numeric_limits<Real>::epsilon()
                   * (abs(scaled) + 1.0);
  return static_cast<Index>(ceil(scaled - eps));
}

void assignVelocityRelativeTolerance(const json&        node,
                                     ConvergenceParams& convergence)
{
  if (node.contains("velocity_relative_tolerance"))
  {
    convergence.velocity_relative_tolerance =
        node.at("velocity_relative_tolerance").get<Real>();
  }
  else if (node.contains("velocity_rel_tol"))
  {
    convergence.velocity_relative_tolerance =
        node.at("velocity_rel_tol").get<Real>();
  }
  else if (node.contains("relative_tolerance"))
  {
    convergence.velocity_relative_tolerance =
        node.at("relative_tolerance").get<Real>();
  }
  else if (node.contains("relative_tol"))
  {
    convergence.velocity_relative_tolerance =
        node.at("relative_tol").get<Real>();
  }
  else if (node.contains("tol"))
  {
    convergence.velocity_relative_tolerance =
        node.at("tol").get<Real>();
  }
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
    throw runtime_error("Config time.convergence must be a boolean or object");
  }

  convergence.enabled = true;
  assign(node, "enabled", convergence.enabled);
  assignVelocityRelativeTolerance(node, convergence);
  assign(node, "min_steps", convergence.min_steps);
}

void parseTimeConfig(const json& node,
                     TimeParams& time)
{
  if (!node.is_object())
  {
    throw runtime_error("Config time must be an object");
  }

  assign(node, "dt", time.dt);
  assign(node, "steps", time.steps);
  if (node.contains("end_time"))
  {
    const Index derived_steps =
        stepsForEndTime(node.at("end_time").get<Real>(), time.dt);
    if (node.contains("steps") && time.steps != derived_steps)
    {
      throw runtime_error(
          "Config time.steps and time.end_time disagree for the configured dt");
    }
    time.steps = derived_steps;
  }
  if (node.contains("convergence"))
  {
    parseConvergenceConfig(node.at("convergence"), time.convergence);
  }
}

void parseVelocityTable(const filesystem::path& path,
                        VelocityParams&         velocity)
{
  ifstream input(path);
  if (!input)
  {
    throw runtime_error("Failed to open velocity table: " + path.string());
  }

  velocity.time.clear();
  velocity.value.clear();

  string line;
  Index  line_no = 0;
  while (getline(input, line))
  {
    ++line_no;
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == string::npos || line[first] == '#')
    {
      continue;
    }

    replace(line.begin(), line.end(), ',', ' ');
    istringstream row(line);
    Real          t = 0.0;
    Real          v = 0.0;
    if (!(row >> t >> v))
    {
      throw runtime_error("Invalid velocity table row "
                          + to_string(line_no) + " in "
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
    prof.type = node.get<string>();
    return prof;
  }
  if (!node.is_object())
  {
    throw runtime_error(
        "Boundary velocity profile must be a string or object");
  }

  assign(node, "type", prof.type);
  assign(node, "radius", prof.rad);
  if (node.contains("center"))
  {
    if (node.at("center").is_string())
    {
      const auto cen = node.at("center").get<string>();
      if (cen != "auto")
      {
        throw runtime_error(
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
  if (node.contains("interpolate"))
  {
    velocity.interp = node.at("interpolate").get<string>();
  }
  else if (node.contains("methodInterpolate"))
  {
    velocity.interp = node.at("methodInterpolate").get<string>();
  }
  else if (node.contains("methodinterpolate"))
  {
    velocity.interp = node.at("methodinterpolate").get<string>();
  }
  else if (node.contains("method"))
  {
    velocity.interp = node.at("method").get<string>();
  }
}

Real parseVelocityScalar(const json&     node,
                         VelocityParams& velocity)
{
  if (node.contains("value"))
  {
    return node.at("value").get<Real>();
  }
  if (node.contains("baseline"))
  {
    return node.at("baseline").get<Real>();
  }
  if (node.contains("bulk_speed"))
  {
    return node.at("bulk_speed").get<Real>();
  }
  if (node.contains("mean_velocity"))
  {
    velocity.qty = "mean_velocity";
    return node.at("mean_velocity").get<Real>();
  }
  if (node.contains("max_velocity"))
  {
    velocity.qty = "max_velocity";
    return node.at("max_velocity").get<Real>();
  }
  if (node.contains("flowrate"))
  {
    velocity.qty = "flowrate";
    return node.at("flowrate").get<Real>();
  }

  throw runtime_error(
      "Boundary velocity time profile requires value, baseline, bulk_speed, mean_velocity, max_velocity, or flowrate");
}

void parseVelocityTimeProfile(const json&             node,
                              const filesystem::path& cfg_dir,
                              VelocityParams&         velocity)
{
  if (!node.is_object())
  {
    throw runtime_error("Boundary velocity time profile must be an object");
  }

  assign(node, "quantity", velocity.qty);

  string type = node.contains("table") ? "table" : "uniform";
  assign(node, "type", type);
  if (type == "table" || type == "csv")
  {
    if (!node.contains("table"))
    {
      throw runtime_error(
          "Boundary velocity table time profile requires table");
    }
    const auto table_path =
        resolveConfigPath(cfg_dir, node.at("table").get<string>());
    parseVelocityTable(table_path, velocity);
    assign(node, "period", velocity.per);
  }
  else if (type == "uniform" || type == "constant" || type == "steady")
  {
    velocity.time  = {0.0};
    velocity.value = {parseVelocityScalar(node, velocity)};
    assign(node, "period", velocity.per);
  }
  else if (type == "sin" || type == "sine")
  {
    Real per = velocity.per > 0.0 ? velocity.per : 1.0;
    assign(node, "period", per);
    if (!isfinite(per) || per <= 0.0)
    {
      throw runtime_error(
          "Boundary velocity sine time profile period must be positive");
    }

    Real amplitude = 0.35;
    assign(node, "amplitude", amplitude);
    assign(node, "pulse_amplitude", amplitude);

    const Real base = parseVelocityScalar(node, velocity);
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
    throw runtime_error(
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

  assign(node, "area", velocity.area);
  assign(node, "quantity", velocity.qty);
  if (node.contains("normal"))
  {
    velocity.nrm =
        parseVector3(node.at("normal"), "Boundary velocity space normal");
  }
}

VelocityParams parseVelocity(const json&             node,
                             const filesystem::path& cfg_dir,
                             const string&           name)
{
  if (!node.is_object())
  {
    throw runtime_error("Boundary " + name + " must be an object");
  }

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
        resolveConfigPath(cfg_dir, node.at("table").get<string>());
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
    throw runtime_error("Boundary " + name + " requires time");
  }
  if (node.contains("value"))
  {
    velocity.value =
        parseRealVector(node.at("value"), "Boundary velocity value");
  }
  else if (!node.contains("table")
           && !(node.contains("time") && node.at("time").is_object()))
  {
    throw runtime_error("Boundary " + name + " requires value");
  }

  if (node.contains("space"))
  {
    parseVelocitySpaceProfile(node.at("space"), velocity);
  }
  assignVelocityInterpolation(node, velocity);

  return velocity;
}

BCsParams parseBoundaryCondition(const json&             node,
                                 const filesystem::path& cfg_dir)
{
  if (!node.is_object())
  {
    throw runtime_error("Boundary condition must be an object");
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
    throw runtime_error(
        "Each boundary condition needs physical, physical_tag, or tag");
  }

  assign(node, "type", cond.type);
  cond.ux = optionalReal(node, "ux");
  cond.uy = optionalReal(node, "uy");
  cond.uz = optionalReal(node, "uz");
  cond.p  = optionalReal(node, "p");
  if (node.contains("velocity") && node.contains("flowrate"))
  {
    throw runtime_error(
        "Boundary condition must not contain both velocity and flowrate");
  }
  if (node.contains("velocity"))
  {
    cond.velocity = parseVelocity(node.at("velocity"), cfg_dir, "velocity");
  }
  else if (node.contains("flowrate"))
  {
    cond.velocity =
        parseVelocity(node.at("flowrate"), cfg_dir, "flowrate");
    cond.velocity->qty = "flowrate";
  }

  if (!cond.ux && !cond.uy && !cond.uz && !cond.p && !cond.velocity)
  {
    throw runtime_error(
        "Boundary condition needs at least one of ux, uy, uz, p, velocity, or flowrate");
  }
  return cond;
}

void validateVelocity(const VelocityParams& velocity)
{
  if (velocity.time.empty())
  {
    throw runtime_error("Boundary velocity time must not be empty");
  }
  if (velocity.time.size() != velocity.value.size())
  {
    throw runtime_error(
        "Boundary velocity time and value must have the same length");
  }
  if (velocity.area <= 0.0)
  {
    throw runtime_error("Boundary velocity area must be positive");
  }
  if (velocity.per < 0.0)
  {
    throw runtime_error("Boundary velocity period must be nonnegative");
  }
  for (Index i = 1; i < velocity.time.size(); ++i)
  {
    if (velocity.time[i] <= velocity.time[i - 1])
    {
      throw runtime_error(
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
    throw runtime_error("Boundary velocity normal must be nonzero");
  }

  if (velocity.interp != "constant" && velocity.interp != "nearest"
      && velocity.interp != "linear" && velocity.interp != "cubic")
  {
    throw runtime_error(
        "Boundary velocity interpolate must be 'constant', 'nearest', 'linear', or 'cubic'");
  }
  if (velocity.qty != "flowrate"
      && velocity.qty != "mean_velocity"
      && velocity.qty != "max_velocity")
  {
    throw runtime_error(
        "Boundary velocity quantity must be 'flowrate', 'mean_velocity', or 'max_velocity'");
  }
  if (velocity.prof.type != "uniform"
      && velocity.prof.type != "poiseuille")
  {
    throw runtime_error(
        "Boundary velocity profile type must be 'uniform' or 'poiseuille'");
  }
  if (velocity.prof.type == "poiseuille" && velocity.prof.rad <= 0.0)
  {
    throw runtime_error(
        "Boundary velocity poiseuille profile requires a positive radius");
  }
}

void validate(const Params& prm)
{
  if (prm.mesh_file.empty())
  {
    throw runtime_error("Config mesh.file is required");
  }
  if (prm.time.steps <= 0 || prm.time.dt <= 0.0)
  {
    throw runtime_error("Time steps and dt must be positive");
  }
  if (prm.time.convergence.enabled)
  {
    if (!isfinite(prm.time.convergence.velocity_relative_tolerance)
        || prm.time.convergence.velocity_relative_tolerance <= 0.0)
    {
      throw runtime_error(
          "Config time.convergence velocity relative tolerance must be positive");
    }
    if (prm.time.convergence.min_steps < 1)
    {
      throw runtime_error("Config time.convergence min_steps must be positive");
    }
  }
  if (prm.fluid.rho <= 0.0 || prm.fluid.mu <= 0.0)
  {
    throw runtime_error("Fluid rho and mu must be positive");
  }
  if (prm.solver.backend != "cpu" && prm.solver.backend != "cuda")
  {
    throw runtime_error("Backend must be either 'cpu' or 'cuda'");
  }
  if (prm.solver.method != "direct"
      && prm.solver.method != "iterative")
  {
    throw runtime_error("Solver method must be either 'direct' or 'iterative'");
  }
  if (prm.solver.solve != "fgmres"
      && prm.solver.solve != "randgmres")
  {
    throw runtime_error("Solver solve must be either 'fgmres' or 'randgmres'");
  }
  if (prm.solver.preconditioner != "ilu0"
      && prm.solver.preconditioner != "none")
  {
    throw runtime_error(
        "Solver preconditioner must be either 'ilu0' or 'none'");
  }
  if (prm.solver.gram_schmidt != "cgs1"
      && prm.solver.gram_schmidt != "cgs2"
      && prm.solver.gram_schmidt != "mgs"
      && prm.solver.gram_schmidt != "mgs_two_sync"
      && prm.solver.gram_schmidt != "mgs_pm")
  {
    throw runtime_error(
        "Solver gram_schmidt must be 'cgs1', 'cgs2', 'mgs', 'mgs_two_sync', or 'mgs_pm'");
  }
  if (prm.solver.sketching != "count" && prm.solver.sketching != "fwht")
  {
    throw runtime_error("Solver sketching must be either 'count' or 'fwht'");
  }
  if (prm.solver.preconditioner_side != "left"
      && prm.solver.preconditioner_side != "right")
  {
    throw runtime_error(
        "Solver preconditioner_side must be either 'left' or 'right'");
  }
  if (prm.solver.flexible && prm.solver.preconditioner_side == "left")
  {
    throw runtime_error(
        "ReSolve flexible GMRES requires right preconditioning");
  }
  if (prm.solver.max_iterations <= 0 || prm.solver.restart <= 0)
  {
    throw runtime_error(
        "Solver max_iterations and restart must be positive");
  }
  if (!isfinite(prm.solver.relative_tolerance)
      || prm.solver.relative_tolerance <= 0.0)
  {
    throw runtime_error("Solver relative_tolerance must be positive");
  }
  if (prm.bcs.empty())
  {
    throw runtime_error("Config bcs must contain at least one boundary");
  }

  for (const auto& cond : prm.bcs)
  {
    if (cond.tag <= 0)
    {
      throw runtime_error(
          "Boundary condition physical tag must be positive");
    }
    if (cond.type != "dirichlet")
    {
      throw runtime_error("Only dirichlet bcs are supported");
    }
    if (cond.velocity)
    {
      validateVelocity(*cond.velocity);
    }
  }
}

Params loadConfig(const string& path)
{
  ifstream input(path);
  if (!input)
  {
    throw runtime_error("Failed to open config file: " + path);
  }

  Params     prm;
  const auto root    = json::parse(input, nullptr, true, true);
  const auto cfg_dir = filesystem::path(path).parent_path();

  if (root.contains("mesh"))
  {
    assign(root.at("mesh"), "file", prm.mesh_file);
  }
  if (root.contains("time"))
  {
    parseTimeConfig(root.at("time"), prm.time);
  }
  if (root.contains("fluid"))
  {
    const auto& fluid = root.at("fluid");
    assign(fluid, "rho", prm.fluid.rho);
    assign(fluid, "mu", prm.fluid.mu);
  }
  if (root.contains("solver"))
  {
    const auto& solver = root.at("solver");
    assign(solver, "backend", prm.solver.backend);
    assign(solver, "method", prm.solver.method);
    if (solver.contains("solve"))
    {
      prm.solver.solve = solver.at("solve").get<string>();
    }
    else if (solver.contains("linear_solver"))
    {
      prm.solver.solve = solver.at("linear_solver").get<string>();
    }
    else if (solver.contains("krylov"))
    {
      prm.solver.solve = solver.at("krylov").get<string>();
    }
    if (solver.contains("preconditioner"))
    {
      prm.solver.preconditioner =
          solver.at("preconditioner").get<string>();
    }
    else if (solver.contains("precond"))
    {
      prm.solver.preconditioner = solver.at("precond").get<string>();
    }
    if (solver.contains("gram_schmidt"))
    {
      prm.solver.gram_schmidt = solver.at("gram_schmidt").get<string>();
    }
    else if (solver.contains("gramSchmidt"))
    {
      prm.solver.gram_schmidt = solver.at("gramSchmidt").get<string>();
    }
    assign(solver, "sketching", prm.solver.sketching);
    if (solver.contains("preconditioner_side"))
    {
      prm.solver.preconditioner_side =
          solver.at("preconditioner_side").get<string>();
    }
    else if (solver.contains("precond_side"))
    {
      prm.solver.preconditioner_side =
          solver.at("precond_side").get<string>();
    }
    assign(solver, "restart", prm.solver.restart);
    if (solver.contains("max_iterations"))
    {
      prm.solver.max_iterations =
          solver.at("max_iterations").get<Index>();
    }
    else if (solver.contains("max_its"))
    {
      prm.solver.max_iterations = solver.at("max_its").get<Index>();
    }
    else if (solver.contains("max_iter"))
    {
      prm.solver.max_iterations = solver.at("max_iter").get<Index>();
    }
    if (solver.contains("relative_tolerance"))
    {
      prm.solver.relative_tolerance =
          solver.at("relative_tolerance").get<Real>();
    }
    else if (solver.contains("rtol"))
    {
      prm.solver.relative_tolerance = solver.at("rtol").get<Real>();
    }
    else if (solver.contains("rel_tol"))
    {
      prm.solver.relative_tolerance = solver.at("rel_tol").get<Real>();
    }
    assign(solver, "flexible", prm.solver.flexible);
  }
  if (root.contains("output"))
  {
    const auto& output = root.at("output");
    assign(output, "interval", prm.output.interval);
    const bool has_directory = output.contains("directory");
    assign(output, "directory", prm.output.directory);
    if (has_directory && !prm.output.directory.empty())
    {
      prm.output.directory =
          resolveConfigPath(cfg_dir, prm.output.directory).string();
    }
  }
  prm.output.interval = max<Index>(1, prm.output.interval);

  if (root.contains("bcs"))
  {
    const auto& boundaries = root.at("bcs");
    if (!boundaries.is_array())
    {
      throw runtime_error("Config bcs must be an array");
    }
    for (const auto& item : boundaries)
    {
      prm.bcs.push_back(parseBoundaryCondition(item, cfg_dir));
    }
  }

  const filesystem::path mesh_path(prm.mesh_file);
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

} // namespace femx
