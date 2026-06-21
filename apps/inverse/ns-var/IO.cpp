#include "IO.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <femx/io/TimeSeriesDataOut.hpp>

namespace femx::navier_var_new
{
namespace
{

BoundarySelector              pressureGauge(const Params& prm);
std::vector<BoundarySelector> fixedVelocityBcs(const Params& prm);

void resizeField(Vector<Real>& out, Index size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

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
    throw std::runtime_error("Config time must be an object");
  }
  assign(node, "dt", time.dt);

  if (node.contains("end_time"))
  {
    time.steps = stepsForEndTime(node.at("end_time").get<Real>(), time.dt);
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
    throw std::runtime_error(name + " must be a tag, name, or object");
  }

  if (node.contains("tag"))
  {
    selector.physical = node.at("tag").get<Index>();
  }
  assign(node, "name", selector.name);
  return selector;
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
    fluid.Re = node.at("reynolds").get<Real>();
  }
  else if (node.contains("reynolds_number"))
  {
    fluid.Re = node.at("reynolds_number").get<Real>();
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

  std::string type = "uniform";
  assign(node, "type", type);
  if (type == "sin" || type == "sine")
  {
    target.type = "poiseuille_pulse";
  }
  else if (type == "uniform" || type == "constant" || type == "steady")
  {
    target.type            = "poiseuille_pulse";
    target.pulse_amplitude = 0.0;
  }
  else
  {
    throw std::runtime_error(
        "Boundary time.type must be 'sin' or 'uniform'");
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

  std::string type = "poiseuille";
  assign(node, "type", type);
  if (type != "poiseuille")
  {
    throw std::runtime_error("Boundary space.type must be 'poiseuille'");
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
  if (!node.contains("velocity") || !node.at("velocity").is_object())
  {
    throw std::runtime_error("Boundary velocity must be an object");
  }

  const auto& velocity_node = node.at("velocity");
  parseTarget(velocity_node, velocity);
  if (velocity_node.contains("time"))
  {
    parseTime(velocity_node.at("time"), velocity);
  }
  if (velocity_node.contains("space"))
  {
    parseSpace(velocity_node.at("space"), velocity);
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
  if (node.contains("tag"))
  {
    cond.physical = node.at("tag").get<Index>();
  }
  else
  {
    throw std::runtime_error("Each boundary condition needs tag");
  }

  assign(node, "type", cond.type);
  cond.ux = optionalReal(node, "ux");
  cond.uy = optionalReal(node, "uy");
  cond.uz = optionalReal(node, "uz");
  cond.p  = optionalReal(node, "p");
  if (node.contains("velocity"))
  {
    cond.vel = parseVelocity(node);
  }

  if (!cond.ux && !cond.uy && !cond.uz && !cond.p && !cond.vel)
  {
    throw std::runtime_error(
        "Forward boundary condition needs at least one of ux, uy, uz, p, or velocity");
  }
  return cond;
}

std::vector<BCsParams> parseBcList(const nlohmann::json& node,
                                   const std::string&    name)
{
  if (!node.is_array())
  {
    throw std::runtime_error("Config " + name + " must be an array");
  }

  std::vector<BCsParams> out;
  out.reserve(node.size());
  for (const auto& item : node)
  {
    out.push_back(parseBc(item));
  }
  return out;
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
    throw std::runtime_error("Config optimizer must be an object");
  }

  assign(node, "type", opt.type);
  if (node.contains("max_iterations"))
  {
    opt.max_iterations = node.at("max_iterations").get<Index>();
  }
  assign(node, "abs_tol", opt.abs_tol);
  assign(node, "rel_tol", opt.rel_tol);
  assign(node, "step_tol", opt.step_tol);
  if (node.contains("scale"))
  {
    const auto& scale = node.at("scale");
    if (!scale.is_object())
    {
      throw std::runtime_error("Config optimizer.scale must be an object");
    }
    assign(scale, "initial_velocity", opt.scale.initial_velocity);
    assign(scale, "boundary", opt.scale.boundary);
  }
}

void parseReg(const nlohmann::json& node,
              RegularizationParams& reg)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config inverse.reg must be an object");
  }
  if (node.contains("beta1"))
  {
    reg.beta1 = node.at("beta1").get<Real>();
  }
  if (node.contains("beta2"))
  {
    reg.beta2 = node.at("beta2").get<Real>();
  }
  if (node.contains("beta3"))
  {
    reg.beta3 = node.at("beta3").get<Real>();
  }
  if (node.contains("beta4"))
  {
    reg.beta4 = node.at("beta4").get<Real>();
  }
}

void parseObjective(const nlohmann::json& node,
                    InverseParams&        inverse)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config objective must be an object");
  }
  assign(node, "alpha", inverse.alpha);
  if (node.contains("reg"))
  {
    parseReg(node.at("reg"), inverse.reg);
  }
}

void parseInitialVelocity(const nlohmann::json&  node,
                          InitialVelocityParams& initial_velocity)
{
  if (!node.is_object())
  {
    throw std::runtime_error(
        "Config controls.initial_velocity must be an object");
  }

  assign(node, "enabled", initial_velocity.enabled);
  if (!node.contains("bounds"))
  {
    return;
  }

  const auto& bounds = node.at("bounds");
  if (!bounds.is_object())
  {
    throw std::runtime_error(
        "Config controls.initial_velocity.bounds must be an object");
  }
  if (bounds.contains("min"))
  {
    initial_velocity.lower = bounds.at("min").get<Real>();
  }
  if (bounds.contains("max"))
  {
    initial_velocity.upper = bounds.at("max").get<Real>();
  }
}

void parseControlBounds(const nlohmann::json& node,
                        BoundsParams&         bounds,
                        const std::string&    name)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config " + name + " must be an object");
  }
  assign(node, "min", bounds.min);
  if (node.contains("max"))
  {
    bounds.max = node.at("max").get<Real>();
  }
  if (node.contains("normal"))
  {
    bounds.normal = parseVector3(node.at("normal"), name + ".normal");
  }
}

void parseControls(const nlohmann::json& node,
                   InverseParams&        inverse)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config controls must be an object");
  }

  if (node.contains("boundary"))
  {
    const auto& boundary = node.at("boundary");
    inverse.ctr          = parseSelector(boundary, "controls.boundary");
    if (boundary.is_object() && boundary.contains("bounds"))
    {
      parseControlBounds(boundary.at("bounds"),
                         inverse.bounds,
                         "controls.boundary.bounds");
    }
  }
  if (node.contains("initial_velocity"))
  {
    inverse.init_vel.enabled = true;
    parseInitialVelocity(node.at("initial_velocity"), inverse.init_vel);
  }
}

void parseInitialGuess(const nlohmann::json& node,
                       InitialGuessParams&   initial_guess)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config initial_guess must be an object");
  }
  if (node.contains("time"))
  {
    parseForwardTime(node.at("time"), initial_guess.time);
    initial_guess.has_time = true;
  }
  if (node.contains("bcs"))
  {
    initial_guess.bcs = parseBcList(node.at("bcs"), "initial_guess.bcs");
  }
}

void parseInv(const nlohmann::json&        node,
              const std::filesystem::path& config_dir,
              InverseParams&               inverse)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config inverse must be an object");
  }

  if (node.contains("controls"))
  {
    parseControls(node.at("controls"), inverse);
  }
  if (node.contains("initial_guess"))
  {
    parseInitialGuess(node.at("initial_guess"), inverse.initial_guess);
  }
  if (node.contains("objective"))
  {
    parseObjective(node.at("objective"), inverse);
  }
  if (node.contains("optimizer"))
  {
    parseOpt(node.at("optimizer"), inverse.opt);
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

bool sameTimeGrid(const TimeParams& lhs,
                  const TimeParams& rhs)
{
  const Real scale = std::max({Real(1.0), std::abs(lhs.dt), std::abs(rhs.dt)});
  const Real tol   = 64.0 * std::numeric_limits<Real>::epsilon() * scale;
  return lhs.steps == rhs.steps && std::abs(lhs.dt - rhs.dt) <= tol;
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
  if (prm.fwd.mesh.file.empty())
  {
    throw std::runtime_error("simulation.mesh.file is required");
  }
  if (prm.fwd.time.steps <= 0 || !std::isfinite(prm.fwd.time.dt)
      || prm.fwd.time.dt <= 0.0)
  {
    throw std::runtime_error("Time steps and dt must be positive");
  }
  if (prm.fwd.fluid.rho <= 0.0)
  {
    throw std::runtime_error("Fluid rho must be positive");
  }
  if (prm.fwd.fluid.mu && *prm.fwd.fluid.mu <= 0.0)
  {
    throw std::runtime_error("Fluid mu must be positive");
  }
  if (prm.fwd.fluid.Re && *prm.fwd.fluid.Re <= 0.0)
  {
    throw std::runtime_error("Fluid reynolds must be positive");
  }
  if (!prm.fwd.fluid.mu && !prm.fwd.fluid.Re)
  {
    throw std::runtime_error("Fluid requires either mu or reynolds");
  }
  if (prm.fwd.bcs.empty())
  {
    throw std::runtime_error("simulation.bcs must contain at least one boundary");
  }
  for (const auto& bc : prm.fwd.bcs)
  {
    if (bc.physical <= 0)
    {
      throw std::runtime_error("simulation.bcs physical tag must be positive");
    }
    if (bc.type != "dirichlet")
    {
      throw std::runtime_error("Only dirichlet simulation.bcs are supported");
    }
    if (bc.vel)
    {
      const auto& velocity = *bc.vel;
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
  validateSelector(prm.inv.ctr, "inverse.ctr");
  const BCsParams& ctr_bc = controlBoundary(prm);
  pressureGauge(prm);
  if (fixedVelocityBcs(prm).empty())
  {
    throw std::runtime_error(
        "simulation.bcs must contain at least one fixed velocity boundary");
  }
  if (prm.fwd.fluid.Re && !ctr_bc.vel)
  {
    throw std::runtime_error(
        "simulation.bcs control velocity profile is required when fluid.Re is used");
  }
  if (prm.fwd.solver.type != "auto"
      && prm.fwd.solver.type != "resolve"
      && prm.fwd.solver.type != "petsc")
  {
    throw std::runtime_error(
        "simulation.solver.type must be 'auto', 'resolve', or 'petsc'");
  }
  if (prm.fwd.solver.backend != "cpu"
      && prm.fwd.solver.backend != "cuda")
  {
    throw std::runtime_error(
        "simulation.solver.backend must be 'cpu' or 'cuda'");
  }
  if (prm.inv.alpha < 0.0
      || prm.inv.reg.beta1 < 0.0
      || prm.inv.reg.beta2 < 0.0
      || prm.inv.reg.beta3 < 0.0
      || prm.inv.reg.beta4 < 0.0)
  {
    throw std::runtime_error(
        "inverse alpha and reg values must be nonnegative");
  }
  if (prm.inv.init_vel.lower
      && prm.inv.init_vel.upper
      && *prm.inv.init_vel.lower
             > *prm.inv.init_vel.upper)
  {
    throw std::runtime_error(
        "inverse.init_vel lower must not exceed upper");
  }
  if (prm.inv.opt.max_iterations < 0
      || prm.inv.opt.abs_tol < 0.0
      || prm.inv.opt.rel_tol < 0.0
      || prm.inv.opt.step_tol < 0.0)
  {
    throw std::runtime_error(
        "inverse opt tolerances and max_iterations must be nonnegative");
  }
  if (!std::isfinite(prm.inv.opt.scale.initial_velocity)
      || prm.inv.opt.scale.initial_velocity <= 0.0
      || !std::isfinite(prm.inv.opt.scale.boundary)
      || prm.inv.opt.scale.boundary <= 0.0)
  {
    throw std::runtime_error("optimizer.scale values must be positive");
  }
  if (prm.inv.opt.type != "lmvm")
  {
    throw std::runtime_error("inverse.opt.type must be 'lmvm'");
  }
  if (prm.inv.bounds.max && *prm.inv.bounds.max < prm.inv.bounds.min)
  {
    throw std::runtime_error(
        "inverse.bounds.max must be greater than min");
  }
  if (!prm.inv.bounds.max && !ctr_bc.vel)
  {
    throw std::runtime_error(
        "inverse.bounds.max is required when the control bcs has no velocity profile");
  }
  if (prm.inv.bounds.max_scale <= 0.0)
  {
    throw std::runtime_error("inverse.bounds.max_scale must be positive");
  }
  Real bounds_normal_norm2 = 0.0;
  for (Real value : prm.inv.bounds.normal)
  {
    bounds_normal_norm2 += value * value;
  }
  if (bounds_normal_norm2 <= 1.0e-28)
  {
    throw std::runtime_error("inverse.bounds.normal must be nonzero");
  }
  if (prm.inv.obs.file.empty())
  {
    throw std::runtime_error("inverse.obs.file is required");
  }
}

Real interpolatedControlParam(const DirichletControl&            control,
                              const Vector<Real>&                prm,
                              const Vector<LinearInterpolation>& stencils,
                              Index                              step,
                              Index                              i)
{
  if (step < 0)
  {
    return 0.0;
  }
  if (step >= stencils.size())
  {
    throw std::runtime_error("control interpolation step is out of range");
  }

  const LinearInterpolation& interp = stencils[step];
  Real                       value  = 0.0;
  interp.forEachWeight(
      [&](Index level, Real weight)
      {
        value += weight * prm[level * control.numDofs() + i];
      });
  return value;
}

void splitState(const MixedFESpace& space,
                const Vector<Real>& x,
                Vector<Real>&       ux,
                Vector<Real>&       uy,
                Vector<Real>&       uz,
                Vector<Real>&       p)
{
  const Index nodes = space.mesh().numNodes();
  resizeField(ux, nodes);
  resizeField(uy, nodes);
  resizeField(uz, nodes);
  resizeField(p, nodes);

  const auto  u_dof = space.field(0);
  const auto  p_dof = space.field(1);
  const Index nc    = u_dof.numComponents();
  for (Index in = 0; in < nodes; ++in)
  {
    ux[in] = x[u_dof.globalDof(in, 0)];
    uy[in] = nc > 1 ? x[u_dof.globalDof(in, 1)] : 0.0;
    uz[in] = nc > 2 ? x[u_dof.globalDof(in, 2)] : 0.0;
    p[in]  = x[p_dof.globalDof(in, 0)];
  }
}

void assignControlDof(const MixedFESpace& space,
                      Index               dof,
                      Real                value,
                      Vector<Real>&       ux,
                      Vector<Real>&       uy,
                      Vector<Real>&       uz)
{
  const auto u_dof = space.field(0);
  for (Index node = 0; node < space.mesh().numNodes(); ++node)
  {
    for (Index d = 0; d < u_dof.numComponents(); ++d)
    {
      if (u_dof.globalDof(node, d) != dof)
      {
        continue;
      }
      if (d == 0)
      {
        ux[node] = value;
      }
      else if (d == 1)
      {
        uy[node] = value;
      }
      else if (d == 2)
      {
        uz[node] = value;
      }
      return;
    }
  }
}

void controlField(const MixedFESpace&                space,
                  const DirichletControl&            control,
                  const Vector<Real>&                prm,
                  const Vector<LinearInterpolation>& stencils,
                  Index                              step,
                  Vector<Real>&                      ux,
                  Vector<Real>&                      uy,
                  Vector<Real>&                      uz)
{
  const Index nodes = space.mesh().numNodes();
  resizeField(ux, nodes);
  resizeField(uy, nodes);
  resizeField(uz, nodes);
  if (step < 0)
  {
    return;
  }

  for (Index i = 0; i < control.numDofs(); ++i)
  {
    assignControlDof(space,
                     control.stateDof(i),
                     interpolatedControlParam(control, prm, stencils, step, i),
                     ux,
                     uy,
                     uz);
  }
}

void ensureDir(const std::string& basename)
{
  const std::filesystem::path path(basename);
  const std::filesystem::path dir = path.parent_path();
  if (!dir.empty())
  {
    std::filesystem::create_directories(dir);
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

  parseForward(root, config_dir, prm.fwd);
  parseInv(root, config_dir, prm.inv);

  if (prm.fwd.bcs.empty() && !prm.inv.initial_guess.bcs.empty())
  {
    prm.fwd.bcs = prm.inv.initial_guess.bcs;
  }
  if (prm.inv.initial_guess.has_time
      && !sameTimeGrid(prm.fwd.time, prm.inv.initial_guess.time))
  {
    throw std::runtime_error(
        "initial_guess.time currently must match the simulation time grid");
  }

  validate(prm);
  return prm;
}

void writeResultViz(const Mesh&                        mesh,
                    const MixedFESpace&                space,
                    const DirichletControl&            control,
                    const solve::TimeTrajectory&       tr,
                    const Vector<Real>&                prm,
                    const Vector<LinearInterpolation>& stencils,
                    Real                               dt,
                    const VizOptions&                  opts,
                    Real                               time_offset)
{
  if (stencils.size() != tr.numSteps())
  {
    throw std::runtime_error(
        "writeResultViz control time stencil size mismatch");
  }
  if (control.numDofs() > 0
      && (prm.empty() || prm.size() % control.numDofs() != 0))
  {
    throw std::runtime_error("writeResultViz control parameter size mismatch");
  }
  const Index ctr_levels =
      control.numDofs() == 0 ? 0 : prm.size() / control.numDofs();
  for (const LinearInterpolation& stencil : stencils)
  {
    if (!stencil.isValid() || stencil.upper >= ctr_levels)
    {
      throw std::runtime_error(
          "writeResultViz control stencil is outside parameter levels");
    }
  }

  ensureDir(opts.basename);

  TimeSeriesDataOut out;
  out.attachMesh(mesh);

  Vector<Real> ux;
  Vector<Real> uy;
  Vector<Real> uz;
  Vector<Real> p;
  Vector<Real> ctr_x;
  Vector<Real> ctr_y;
  Vector<Real> ctr_z;

  for (Index level = 0; level < tr.numLevels(); ++level)
  {
    out.beginStep(time_offset + static_cast<Real>(level) * dt);

    splitState(space, tr[level], ux, uy, uz, p);
    out.addNodalVectorField("velocity", ux, uy, uz);
    out.addNodalScalarField("pressure", p);

    const Index step = level == 0 ? -1 : level - 1;
    controlField(space, control, prm, stencils, step, ctr_x, ctr_y, ctr_z);
    out.addNodalVectorField("control", ctr_x, ctr_y, ctr_z);
  }

  out.write(opts.basename);
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
  return bc.ux || bc.uy || bc.uz || bc.vel;
}

Real reLength(const TargetParams& target)
{
  return 2.0 * target.radius;
}

BoundarySelector pressureGauge(const Params& prm)
{
  for (const auto& bc : prm.fwd.bcs)
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
  for (const auto& bc : prm.fwd.bcs)
  {
    if (matches(prm.inv.ctr, bc) || !hasFixedVel(bc))
    {
      continue;
    }
    selectors.push_back(bcSelector(bc));
  }
  return selectors;
}

} // namespace

const BCsParams& controlBoundary(const Params& prm)
{
  for (const auto& bc : prm.fwd.bcs)
  {
    if (matches(prm.inv.ctr, bc))
    {
      return bc;
    }
  }
  throw std::runtime_error("inverse.ctr did not match any simulation.bcs");
}

const TargetParams& controlTarget(const Params& prm)
{
  const BCsParams& bc = controlBoundary(prm);
  if (!bc.vel)
  {
    throw std::runtime_error(
        "inverse.ctr must refer to a simulation.bcs entry with velocity");
  }
  return *bc.vel;
}

FluidParams fluidParams(const Params& prm)
{
  const auto& config = prm.fwd.fluid;

  FluidParams fluid;
  fluid.rho = config.rho;
  if (config.mu)
  {
    fluid.mu = *config.mu;
    return fluid;
  }

  const TargetParams& target = controlTarget(prm);
  fluid.mu                   = config.rho * target.bulk_speed * reLength(target)
                               / *config.Re;
  return fluid;
}

} // namespace femx::navier_var_new
