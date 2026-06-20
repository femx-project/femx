#include "Config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace femx::navier_var
{
namespace
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
    values[static_cast<std::size_t>(i)] = node.at(i).get<Real>();
  }
  return values;
}

std::array<Index, 3> parseIndex3(const nlohmann::json& node,
                                 const std::string&    name)
{
  if (!node.is_array() || node.size() != 3)
  {
    throw std::runtime_error(name + " must be an array with 3 values");
  }

  std::array<Index, 3> values{};
  for (Index i = 0; i < 3; ++i)
  {
    values[static_cast<std::size_t>(i)] = node.at(i).get<Index>();
  }
  return values;
}

std::vector<Index> parseIndexList(const nlohmann::json& node,
                                  const std::string&    name)
{
  if (!node.is_array())
  {
    throw std::runtime_error(name + " must be an array");
  }

  std::vector<Index> values;
  values.reserve(node.size());
  for (const auto& item : node)
  {
    values.push_back(item.get<Index>());
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

Index stepsForEndTime(Real end_time,
                      Real dt)
{
  if (!std::isfinite(end_time) || end_time <= 0.0)
  {
    throw std::runtime_error("simulation.time.end_time must be positive");
  }
  if (!std::isfinite(dt) || dt <= 0.0)
  {
    throw std::runtime_error("simulation.time.dt must be positive");
  }

  const Real scaled = end_time / dt;
  if (!std::isfinite(scaled) || scaled <= 0.0
      || scaled > static_cast<Real>(std::numeric_limits<Index>::max()))
  {
    throw std::runtime_error("simulation.time.end_time / dt is out of range");
  }

  const Real eps = 64.0 * std::numeric_limits<Real>::epsilon()
                   * (std::abs(scaled) + 1.0);
  return static_cast<Index>(std::ceil(scaled - eps));
}

void parseForwardTime(const nlohmann::json& node,
                      TimeParams&           time)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config simulation.time must be an object");
  }

  const bool has_steps = node.contains("steps")
                         || node.contains("num_steps");
  if (node.contains("steps"))
  {
    time.steps = node.at("steps").get<Index>();
  }
  else if (node.contains("num_steps"))
  {
    time.steps = node.at("num_steps").get<Index>();
  }
  assign(node, "dt", time.dt);

  std::optional<Real> end_time;
  if (node.contains("end_time"))
  {
    end_time = node.at("end_time").get<Real>();
  }
  else if (node.contains("finish_time"))
  {
    end_time = node.at("finish_time").get<Real>();
  }
  else if (node.contains("t_end"))
  {
    end_time = node.at("t_end").get<Real>();
  }
  else if (node.contains("duration"))
  {
    end_time = node.at("duration").get<Real>();
  }

  if (end_time)
  {
    const Index derived_steps = stepsForEndTime(*end_time, time.dt);
    if (has_steps && time.steps != derived_steps)
    {
      throw std::runtime_error(
          "simulation.time steps and end_time disagree for the configured dt");
    }
    time.steps = derived_steps;
  }
}

void parseOutput(const nlohmann::json& node,
                 OutputParams&         output);

void parseSolver(const nlohmann::json& node,
                 SolverParams&         solver);

void parseObsGrid(const nlohmann::json&    node,
                  ObservationParams::Grid& grid)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config inverse.obs.grid must be an object");
  }

  if (node.contains("counts"))
  {
    grid.counts = parseIndex3(node.at("counts"), "inverse.obs.grid.counts");
  }
  else if (node.contains("shape"))
  {
    grid.counts = parseIndex3(node.at("shape"), "inverse.obs.grid.shape");
  }

  if (node.contains("bounds"))
  {
    const auto& bounds = node.at("bounds");
    if (!bounds.is_array() || bounds.size() != 2)
    {
      throw std::runtime_error(
          "Config inverse.obs.grid.bounds must contain lower and upper points");
    }
    grid.lower = parseVector3(bounds.at(0), "inverse.obs.grid.bounds[0]");
    grid.upper = parseVector3(bounds.at(1), "inverse.obs.grid.bounds[1]");
  }
  if (node.contains("lower"))
  {
    grid.lower = parseVector3(node.at("lower"), "inverse.obs.grid.lower");
  }
  else if (node.contains("min"))
  {
    grid.lower = parseVector3(node.at("min"), "inverse.obs.grid.min");
  }
  if (node.contains("upper"))
  {
    grid.upper = parseVector3(node.at("upper"), "inverse.obs.grid.upper");
  }
  else if (node.contains("max"))
  {
    grid.upper = parseVector3(node.at("max"), "inverse.obs.grid.max");
  }

  if (node.contains("origin"))
  {
    grid.origin      = parseVector3(node.at("origin"), "inverse.obs.grid.origin");
    grid.use_spacing = true;
  }
  if (node.contains("spacing"))
  {
    grid.spacing     = parseVector3(node.at("spacing"), "inverse.obs.grid.spacing");
    grid.use_spacing = true;
  }
}

void parseObs(const nlohmann::json&        node,
              const std::filesystem::path& config_dir,
              ObservationParams&           obs)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config inverse.obs must be an object");
  }

  if (node.contains("type"))
  {
    const std::string type = node.at("type").get<std::string>();
    if (type != "grid")
    {
      throw std::runtime_error("Config inverse.obs.type must be 'grid'");
    }
    obs.type = "grid";
  }

  const bool has_file = node.contains("file") || node.contains("path")
                        || node.contains("data_file")
                        || node.contains("data");
  if (node.contains("file"))
  {
    obs.file = node.at("file").get<std::string>();
  }
  else if (node.contains("path"))
  {
    obs.file = node.at("path").get<std::string>();
  }
  else if (node.contains("data_file"))
  {
    obs.file = node.at("data_file").get<std::string>();
  }
  else if (node.contains("data"))
  {
    obs.file = node.at("data").get<std::string>();
  }
  if (has_file)
  {
    obs.file = resolveConfigPath(config_dir, obs.file).string();
  }

  if (node.contains("components"))
  {
    obs.components = parseIndexList(node.at("components"),
                                    "inverse.obs.components");
  }

  if (node.contains("grid") || node.contains("counts"))
  {
    obs.grid = ObservationParams::Grid{};
    if (node.contains("grid"))
    {
      parseObsGrid(node.at("grid"), *obs.grid);
    }
    else
    {
      parseObsGrid(node, *obs.grid);
    }
    obs.type = "grid";
  }
}

BoundarySelector parseSelector(const nlohmann::json& node,
                               const std::string&    name)
{
  BoundarySelector selector;
  if (node.is_number_integer())
  {
    selector.physical = node.get<Index>();
    return selector;
  }
  if (node.is_string())
  {
    selector.name = node.get<std::string>();
    return selector;
  }
  if (!node.is_object())
  {
    throw std::runtime_error(name + " must be a physical tag, name, or object");
  }

  if (node.contains("physical"))
  {
    selector.physical = node.at("physical").get<Index>();
  }
  else if (node.contains("physical_tag"))
  {
    selector.physical = node.at("physical_tag").get<Index>();
  }
  else if (node.contains("tag"))
  {
    selector.physical = node.at("tag").get<Index>();
  }
  assign(node, "name", selector.name);
  return selector;
}

void parseSelectors(const nlohmann::json&          node,
                    const std::string&             name,
                    std::vector<BoundarySelector>& out)
{
  out.clear();
  if (!node.is_array())
  {
    out.push_back(parseSelector(node, name));
    return;
  }

  for (const auto& item : node)
  {
    out.push_back(parseSelector(item, name));
  }
}

void parseMesh(const nlohmann::json&        node,
               const std::filesystem::path& config_dir,
               MeshParams&                  mesh)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config mesh must be an object");
  }

  assign(node, "file", mesh.file);

  if (!mesh.file.empty())
  {
    mesh.file = resolveConfigPath(config_dir, mesh.file).string();
  }
}

void parseFluid(const nlohmann::json& node,
                FluidConfig&          fluid)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config fluid must be an object");
  }

  assign(node, "rho", fluid.rho);
  if (node.contains("mu"))
  {
    fluid.mu = node.at("mu").get<Real>();
  }
  if (node.contains("reynolds"))
  {
    fluid.reynolds = node.at("reynolds").get<Real>();
  }
  else if (node.contains("reynolds_number"))
  {
    fluid.reynolds = node.at("reynolds_number").get<Real>();
  }
}

void parseTarget(const nlohmann::json& node,
                 TargetParams&         target)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary velocity must be an object");
  }

  assign(node, "type", target.type);
  assign(node, "quantity", target.quantity);
  assign(node, "bulk_speed", target.bulk_speed);
  if (node.contains("mean_velocity"))
  {
    target.bulk_speed = node.at("mean_velocity").get<Real>();
    target.quantity   = "mean_velocity";
  }
  else if (node.contains("max_velocity"))
  {
    target.bulk_speed = node.at("max_velocity").get<Real>();
    target.quantity   = "max_velocity";
  }
  else if (node.contains("value"))
  {
    target.bulk_speed = node.at("value").get<Real>();
  }
  else if (node.contains("baseline"))
  {
    target.bulk_speed = node.at("baseline").get<Real>();
  }
  assign(node, "pulse_amplitude", target.pulse_amplitude);
  if (node.contains("amplitude"))
  {
    target.pulse_amplitude = node.at("amplitude").get<Real>();
  }
  assign(node, "period", target.period);
  assign(node, "radius", target.radius);
  if (node.contains("center"))
  {
    target.center = parseVector3(node.at("center"), "simulation.center");
  }
  if (node.contains("normal"))
  {
    target.normal = parseVector3(node.at("normal"), "simulation.normal");
  }
}

void parseTime(const nlohmann::json& node,
               TargetParams&         target)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary time profile must be an object");
  }

  std::string profile = "sin";
  assign(node, "profile", profile);
  if (profile == "sin" || profile == "sine")
  {
    target.type = "poiseuille_pulse";
  }
  else
  {
    throw std::runtime_error("Boundary time.profile must be 'sin'");
  }

  if (node.contains("value"))
  {
    target.bulk_speed = node.at("value").get<Real>();
  }
  else if (node.contains("baseline"))
  {
    target.bulk_speed = node.at("baseline").get<Real>();
  }
  else if (node.contains("bulk_speed"))
  {
    target.bulk_speed = node.at("bulk_speed").get<Real>();
  }
  else if (node.contains("mean_velocity"))
  {
    target.bulk_speed = node.at("mean_velocity").get<Real>();
    target.quantity   = "mean_velocity";
  }
  else if (node.contains("max_velocity"))
  {
    target.bulk_speed = node.at("max_velocity").get<Real>();
    target.quantity   = "max_velocity";
  }

  if (node.contains("amplitude"))
  {
    target.pulse_amplitude = node.at("amplitude").get<Real>();
  }
  else if (node.contains("pulse_amplitude"))
  {
    target.pulse_amplitude = node.at("pulse_amplitude").get<Real>();
  }
  assign(node, "period", target.period);
}

void parseSpace(const nlohmann::json& node,
                TargetParams&         target)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Boundary space profile must be an object");
  }

  std::string profile = "poiseuille";
  if (node.contains("profile"))
  {
    profile = node.at("profile").get<std::string>();
  }
  else if (node.contains("type"))
  {
    profile = node.at("type").get<std::string>();
  }
  if (profile != "poiseuille")
  {
    throw std::runtime_error("Boundary space.profile must be 'poiseuille'");
  }

  target.type = "poiseuille_pulse";
  assign(node, "quantity", target.quantity);
  assign(node, "radius", target.radius);
  if (node.contains("center"))
  {
    target.center =
        parseVector3(node.at("center"), "simulation.bcs.space.center");
  }
  if (node.contains("normal"))
  {
    target.normal =
        parseVector3(node.at("normal"), "simulation.bcs.space.normal");
  }
}

TargetParams parseVelocity(const nlohmann::json& node)
{
  TargetParams velocity;
  if (node.contains("velocity"))
  {
    parseTarget(node.at("velocity"), velocity);
  }
  if (node.contains("time"))
  {
    parseTime(node.at("time"), velocity);
  }
  if (node.contains("space"))
  {
    parseSpace(node.at("space"), velocity);
  }
  return velocity;
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
    throw std::runtime_error(std::string("Boundary value ") + key
                             + " must be a number");
  }
  return node.at(key).get<Real>();
}

BCsParams parseBc(const nlohmann::json& node)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Forward boundary condition must be an object");
  }

  BCsParams cond;
  assign(node, "name", cond.name);
  if (node.contains("physical"))
  {
    cond.physical = node.at("physical").get<Index>();
  }
  else if (node.contains("physical_tag"))
  {
    cond.physical = node.at("physical_tag").get<Index>();
  }
  else if (node.contains("tag"))
  {
    cond.physical = node.at("tag").get<Index>();
  }
  else
  {
    throw std::runtime_error(
        "Each forward boundary condition needs physical, physical_tag, or tag");
  }

  assign(node, "type", cond.type);
  cond.ux = optionalReal(node, "ux");
  cond.uy = optionalReal(node, "uy");
  cond.uz = optionalReal(node, "uz");
  cond.p  = optionalReal(node, "p");
  if (node.contains("velocity") || node.contains("time") || node.contains("space"))
  {
    cond.velocity = parseVelocity(node);
  }

  if (!cond.ux && !cond.uy && !cond.uz && !cond.p && !cond.velocity)
  {
    throw std::runtime_error(
        "Forward boundary condition needs at least one of ux, uy, uz, p, or velocity");
  }
  return cond;
}

void parseBcs(const nlohmann::json& node,
              ForwardParams&        forward)
{
  if (!node.is_array())
  {
    throw std::runtime_error("Config simulation.bcs must be an array");
  }

  forward.bcs.clear();
  for (const auto& item : node)
  {
    forward.bcs.push_back(parseBc(item));
  }
}

void parseForward(const nlohmann::json&        node,
                  const std::filesystem::path& config_dir,
                  ForwardParams&               forward)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config simulation must be an object");
  }

  if (node.contains("mesh"))
  {
    parseMesh(node.at("mesh"), config_dir, forward.mesh);
  }
  if (node.contains("time"))
  {
    parseForwardTime(node.at("time"), forward.time);
  }
  if (node.contains("fluid"))
  {
    parseFluid(node.at("fluid"), forward.fluid);
  }
  if (node.contains("bcs"))
  {
    parseBcs(node.at("bcs"), forward);
  }
  if (node.contains("output"))
  {
    parseOutput(node.at("output"), forward.output);
  }
  if (node.contains("solver"))
  {
    parseSolver(node.at("solver"), forward.solver);
  }
}

void parseOpt(const nlohmann::json& node,
              OptimizerParams&      opt)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config inverse.opt must be an object");
  }

  assign(node, "type", opt.type);
  if (node.contains("max_iterations"))
  {
    opt.max_iterations = node.at("max_iterations").get<Index>();
  }
  else if (node.contains("max_its"))
  {
    opt.max_iterations = node.at("max_its").get<Index>();
  }
  assign(node, "grad_abs_tolerance", opt.grad_abs_tolerance);
  assign(node, "grad_rel_tolerance", opt.grad_rel_tolerance);
  assign(node, "grad_step_tolerance", opt.grad_step_tolerance);
  assign(node, "use_options_database", opt.use_options_database);
}

void parseInitialVelocity(const nlohmann::json&  node,
                          InitialVelocityParams& initial_velocity)
{
  if (node.is_boolean())
  {
    initial_velocity.enabled = node.get<bool>();
    return;
  }
  if (!node.is_object())
  {
    throw std::runtime_error(
        "Config inverse.initial_velocity must be an object or boolean");
  }

  assign(node, "enabled", initial_velocity.enabled);
  assign(node, "search", initial_velocity.enabled);
  if (node.contains("lower"))
  {
    initial_velocity.lower = node.at("lower").get<Real>();
  }
  else if (node.contains("min"))
  {
    initial_velocity.lower = node.at("min").get<Real>();
  }
  if (node.contains("upper"))
  {
    initial_velocity.upper = node.at("upper").get<Real>();
  }
  else if (node.contains("max"))
  {
    initial_velocity.upper = node.at("max").get<Real>();
  }
  assign(node, "l2", initial_velocity.l2);
}

void parseInv(const nlohmann::json&        node,
              const std::filesystem::path& config_dir,
              InverseParams&               inverse)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config inverse must be an object");
  }

  if (node.contains("control"))
  {
    inverse.control = parseSelector(node.at("control"), "inverse.control");
  }
  assign(node, "alpha", inverse.alpha);
  if (node.contains("reg"))
  {
    const auto& reg = node.at("reg");
    if (!reg.is_object())
    {
      throw std::runtime_error("Config inverse.reg must be an object");
    }
    if (reg.contains("time"))
    {
      inverse.reg.time = reg.at("time").get<Real>();
    }
    if (reg.contains("l2"))
    {
      inverse.reg.l2 = reg.at("l2").get<Real>();
    }
  }
  if (node.contains("opt"))
  {
    parseOpt(node.at("opt"), inverse.opt);
  }
  if (node.contains("bounds"))
  {
    const auto& bounds = node.at("bounds");
    if (!bounds.is_object())
    {
      throw std::runtime_error("Config inverse.bounds must be an object");
    }
    assign(bounds, "axial_min", inverse.bounds.axial_min);
    if (bounds.contains("axial_max"))
    {
      inverse.bounds.axial_max = bounds.at("axial_max").get<Real>();
    }
    assign(bounds, "axial_max_scale", inverse.bounds.axial_max_scale);
    assign(bounds, "fix_non_axial", inverse.bounds.fix_non_axial);
    if (bounds.contains("normal"))
    {
      inverse.bounds.normal =
          parseVector3(bounds.at("normal"), "inverse.bounds.normal");
    }
  }
  if (node.contains("initial_velocity"))
  {
    parseInitialVelocity(node.at("initial_velocity"),
                         inverse.initial_velocity);
  }
  if (node.contains("initial_condition"))
  {
    const auto& initial = node.at("initial_condition");
    if (!initial.is_object())
    {
      throw std::runtime_error(
          "Config inverse.initial_condition must be an object");
    }
    if (initial.contains("velocity"))
    {
      parseInitialVelocity(initial.at("velocity"),
                           inverse.initial_velocity);
    }
  }
  if (node.contains("obs"))
  {
    parseObs(node.at("obs"), config_dir, inverse.obs);
  }
}

void parseOutput(const nlohmann::json& node,
                 OutputParams&         output)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config output must be an object");
  }

  assign(node, "basename", output.basename);
}

void parseSolver(const nlohmann::json& node,
                 SolverParams&         solver)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config solver must be an object");
  }
  assign(node, "type", solver.type);
  assign(node, "backend", solver.backend);
}

bool hasSelector(const BoundarySelector& selector)
{
  return selector.physical > 0 || !selector.name.empty();
}

void validateSelector(const BoundarySelector& selector,
                      const std::string&      name)
{
  if (!hasSelector(selector))
  {
    throw std::runtime_error(name + " requires a physical tag or name");
  }
}

void validate(const Params& prm)
{
  if (prm.forward.mesh.file.empty())
  {
    throw std::runtime_error("simulation.mesh.file is required");
  }
  if (prm.forward.time.steps <= 0 || !std::isfinite(prm.forward.time.dt)
      || prm.forward.time.dt <= 0.0)
  {
    throw std::runtime_error("Time steps and dt must be positive");
  }
  if (prm.forward.fluid.rho <= 0.0)
  {
    throw std::runtime_error("Fluid rho must be positive");
  }
  if (prm.forward.fluid.mu && *prm.forward.fluid.mu <= 0.0)
  {
    throw std::runtime_error("Fluid mu must be positive");
  }
  if (prm.forward.fluid.reynolds && *prm.forward.fluid.reynolds <= 0.0)
  {
    throw std::runtime_error("Fluid reynolds must be positive");
  }
  if (!prm.forward.fluid.mu && !prm.forward.fluid.reynolds)
  {
    throw std::runtime_error("Fluid requires either mu or reynolds");
  }
  if (prm.forward.bcs.empty())
  {
    throw std::runtime_error("simulation.bcs must contain at least one boundary");
  }
  for (const auto& bc : prm.forward.bcs)
  {
    if (bc.physical <= 0)
    {
      throw std::runtime_error("simulation.bcs physical tag must be positive");
    }
    if (bc.type != "dirichlet")
    {
      throw std::runtime_error("Only dirichlet simulation.bcs are supported");
    }
    if (bc.velocity)
    {
      const auto& velocity = *bc.velocity;
      if (velocity.type != "poiseuille_pulse")
      {
        throw std::runtime_error(
            "simulation.bcs velocity.type must be 'poiseuille_pulse'");
      }
      if (velocity.bulk_speed <= 0.0 || velocity.period <= 0.0
          || velocity.radius <= 0.0)
      {
        throw std::runtime_error(
            "simulation.bcs velocity bulk_speed, period, and radius must be positive");
      }
      if (velocity.quantity != "mean_velocity"
          && velocity.quantity != "bulk_speed"
          && velocity.quantity != "max_velocity")
      {
        throw std::runtime_error(
            "simulation.bcs velocity quantity must be 'mean_velocity', 'bulk_speed', or 'max_velocity'");
      }
      Real normal_norm2 = 0.0;
      for (Real value : velocity.normal)
      {
        normal_norm2 += value * value;
      }
      if (normal_norm2 <= 1.0e-28)
      {
        throw std::runtime_error("simulation.bcs velocity.normal must be nonzero");
      }
    }
  }
  validateSelector(prm.inverse.control, "inverse.control");
  const BCsParams& control_bc = controlBoundary(prm);
  pressureGauge(prm);
  if (fixedVelocityBcs(prm).empty())
  {
    throw std::runtime_error(
        "simulation.bcs must contain at least one fixed velocity boundary");
  }
  if (prm.forward.fluid.reynolds && !control_bc.velocity)
  {
    throw std::runtime_error(
        "simulation.bcs control velocity profile is required when fluid.reynolds is used");
  }
  if (prm.forward.solver.type != "auto"
      && prm.forward.solver.type != "resolve"
      && prm.forward.solver.type != "petsc")
  {
    throw std::runtime_error(
        "simulation.solver.type must be 'auto', 'resolve', or 'petsc'");
  }
  if (prm.forward.solver.backend != "cpu"
      && prm.forward.solver.backend != "cuda")
  {
    throw std::runtime_error(
        "simulation.solver.backend must be 'cpu' or 'cuda'");
  }
  if (prm.inverse.alpha < 0.0
      || prm.inverse.reg.time < 0.0
      || prm.inverse.reg.l2 < 0.0
      || prm.inverse.initial_velocity.l2 < 0.0)
  {
    throw std::runtime_error(
        "inverse alpha and reg values must be nonnegative");
  }
  if (prm.inverse.initial_velocity.lower
      && prm.inverse.initial_velocity.upper
      && *prm.inverse.initial_velocity.lower
             > *prm.inverse.initial_velocity.upper)
  {
    throw std::runtime_error(
        "inverse.initial_velocity lower must not exceed upper");
  }
  if (prm.inverse.opt.max_iterations < 0
      || prm.inverse.opt.grad_abs_tolerance < 0.0
      || prm.inverse.opt.grad_rel_tolerance < 0.0
      || prm.inverse.opt.grad_step_tolerance < 0.0)
  {
    throw std::runtime_error(
        "inverse opt tolerances and max_iterations must be nonnegative");
  }
  if (prm.inverse.opt.type != "lmvm")
  {
    throw std::runtime_error("inverse.opt.type must be 'lmvm'");
  }
  if (prm.inverse.bounds.axial_max
      && *prm.inverse.bounds.axial_max < prm.inverse.bounds.axial_min)
  {
    throw std::runtime_error(
        "inverse.bounds.axial_max must be greater than axial_min");
  }
  if (!prm.inverse.bounds.axial_max && !control_bc.velocity)
  {
    throw std::runtime_error(
        "inverse.bounds.axial_max is required when the control bcs has no velocity profile");
  }
  if (prm.inverse.bounds.axial_max_scale <= 0.0)
  {
    throw std::runtime_error("inverse.bounds.axial_max_scale must be positive");
  }
  Real bounds_normal_norm2 = 0.0;
  for (Real value : prm.inverse.bounds.normal)
  {
    bounds_normal_norm2 += value * value;
  }
  if (bounds_normal_norm2 <= 1.0e-28)
  {
    throw std::runtime_error("inverse.bounds.normal must be nonzero");
  }
  if (prm.inverse.obs.file.empty())
  {
    throw std::runtime_error("inverse.obs.file is required");
  }
}

} // namespace

Params loadConfig(const std::string& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::runtime_error("Failed to open config file: " + path);
  }

  Params     prm;
  const auto root       = nlohmann::json::parse(input, nullptr, true, true);
  const auto config_dir = std::filesystem::path(path).parent_path();
  if (!root.is_object())
  {
    throw std::runtime_error("Config root must be an object");
  }

  if (root.contains("simulation") && root.contains("forward"))
  {
    throw std::runtime_error(
        "Config must not contain both simulation and legacy forward sections");
  }
  if (root.contains("simulation"))
  {
    parseForward(root.at("simulation"), config_dir, prm.forward);
  }
  else if (root.contains("forward"))
  {
    parseForward(root.at("forward"), config_dir, prm.forward);
  }
  if (root.contains("inverse"))
  {
    parseInv(root.at("inverse"), config_dir, prm.inverse);
  }

  validate(prm);
  return prm;
}

BoundarySelector bcSelector(const BCsParams& bc)
{
  return BoundarySelector{bc.physical, bc.name};
}

namespace
{

bool matches(const BoundarySelector& selector,
             const BCsParams&        bc)
{
  if (!selector.name.empty())
  {
    return bc.name == selector.name;
  }
  return bc.physical == selector.physical;
}

bool hasFixedVel(const BCsParams& bc)
{
  return bc.ux || bc.uy || bc.uz || bc.velocity;
}

} // namespace

const BCsParams& controlBoundary(const Params& prm)
{
  for (const auto& bc : prm.forward.bcs)
  {
    if (matches(prm.inverse.control, bc))
    {
      return bc;
    }
  }
  throw std::runtime_error("inverse.control did not match any simulation.bcs");
}

const TargetParams& controlTarget(const Params& prm)
{
  const BCsParams& bc = controlBoundary(prm);
  if (!bc.velocity)
  {
    throw std::runtime_error(
        "inverse.control must refer to a simulation.bcs entry with velocity");
  }
  return *bc.velocity;
}

Real reLength(const TargetParams& target)
{
  return 2.0 * target.radius;
}

BoundarySelector pressureGauge(const Params& prm)
{
  for (const auto& bc : prm.forward.bcs)
  {
    if (bc.p)
    {
      return bcSelector(bc);
    }
  }
  throw std::runtime_error("simulation.bcs must contain a pressure boundary");
}

std::vector<BoundarySelector> fixedVelocityBcs(const Params& prm)
{
  std::vector<BoundarySelector> selectors;
  for (const auto& bc : prm.forward.bcs)
  {
    if (matches(prm.inverse.control, bc) || !hasFixedVel(bc))
    {
      continue;
    }
    selectors.push_back(bcSelector(bc));
  }
  return selectors;
}

FluidParams fluidParams(const Params& prm)
{
  const auto& config = prm.forward.fluid;

  FluidParams fluid;
  fluid.rho = config.rho;
  if (config.mu)
  {
    fluid.mu = *config.mu;
    return fluid;
  }

  const TargetParams& target = controlTarget(prm);
  fluid.mu                   = config.rho * target.bulk_speed * reLength(target)
             / *config.reynolds;
  return fluid;
}

Real Re(const Params& prm)
{
  const FluidParams fluid  = fluidParams(prm);
  const auto&       target = controlTarget(prm);
  return fluid.rho * target.bulk_speed * reLength(target) / fluid.mu;
}

} // namespace femx::navier_var
