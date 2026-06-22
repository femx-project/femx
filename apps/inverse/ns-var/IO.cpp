#include "IO.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/linalg/BlockVectorView.hpp>

using namespace std;
using nlohmann::json;
using namespace femx::state;

namespace femx::navier_var_new
{
namespace
{

BoundarySelector         pressureGauge(const Params& prm);
Vector<BoundarySelector> fixedVelocityBcs(const Params& prm);

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

array<Index, 3> parseIndex3(const json&   node,
                            const string& name)
{
  if (!node.is_array() || node.size() != 3)
  {
    throw runtime_error(name + " must be an array with 3 values");
  }

  array<Index, 3> vals{};
  for (Index i = 0; i < 3; ++i)
  {
    vals[i] = node.at(i).get<Index>();
  }
  return vals;
}

Vector<Index> parseIndexList(const json&   node,
                             const string& name)
{
  if (!node.is_array())
  {
    throw runtime_error(name + " must be an array");
  }

  Vector<Index> vals;
  for (const auto& item : node)
  {
    vals.push_back(item.get<Index>());
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
    throw runtime_error("simulation.time.end_time must be positive");
  }
  if (!isfinite(dt) || dt <= 0.0)
  {
    throw runtime_error("simulation.time.dt must be positive");
  }

  const Real scaled = end_time / dt;
  if (!isfinite(scaled) || scaled <= 0.0
      || scaled > static_cast<Real>(numeric_limits<Index>::max()))
  {
    throw runtime_error("simulation.time.end_time / dt is out of range");
  }

  const Real eps = 64.0 * numeric_limits<Real>::epsilon()
                   * (abs(scaled) + 1.0);
  return static_cast<Index>(ceil(scaled - eps));
}

void parseForwardTime(const json& node,
                      TimeParams& time)
{
  if (!node.is_object())
  {
    throw runtime_error("Config time must be an object");
  }
  const bool has_steps = node.contains("steps")
                         || node.contains("num_steps");
  assign(node, "dt", time.dt);
  if (node.contains("steps"))
  {
    time.steps = node.at("steps").get<Index>();
  }
  else if (node.contains("num_steps"))
  {
    time.steps = node.at("num_steps").get<Index>();
  }

  if (node.contains("end_time"))
  {
    const Index derived_steps =
        stepsForEndTime(node.at("end_time").get<Real>(), time.dt);
    if (has_steps && time.steps != derived_steps)
    {
      throw runtime_error(
          "Config time.steps and time.end_time disagree for the configured dt");
    }
    time.steps = derived_steps;
  }
}

void parseOutput(const json&   node,
                 OutputParams& output);

void parseSolver(const json&   node,
                 SolverParams& solver);

void parseObsGrid(const json&              node,
                  ObservationParams::Grid& grid)
{
  if (!node.is_object())
  {
    throw runtime_error("Config inverse.obs.grid must be an object");
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
    const auto& bnds = node.at("bounds");
    if (!bnds.is_array() || bnds.size() != 2)
    {
      throw runtime_error(
          "Config inverse.obs.grid.bounds must contain lower and upper points");
    }
    grid.lower = parseVector3(bnds.at(0), "inverse.obs.grid.bounds[0]");
    grid.upper = parseVector3(bnds.at(1), "inverse.obs.grid.bounds[1]");
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

void parseObs(const json&             node,
              const filesystem::path& cfg_dir,
              ObservationParams&      obs)
{
  if (!node.is_object())
  {
    throw runtime_error("Config inverse.obs must be an object");
  }

  if (node.contains("type"))
  {
    const string type = node.at("type").get<string>();
    if (type != "grid")
    {
      throw runtime_error("Config inverse.obs.type must be 'grid'");
    }
    obs.type = "grid";
  }

  const bool has_file = node.contains("file") || node.contains("path")
                        || node.contains("data_file")
                        || node.contains("data");
  if (node.contains("file"))
  {
    obs.file = node.at("file").get<string>();
  }
  else if (node.contains("path"))
  {
    obs.file = node.at("path").get<string>();
  }
  else if (node.contains("data_file"))
  {
    obs.file = node.at("data_file").get<string>();
  }
  else if (node.contains("data"))
  {
    obs.file = node.at("data").get<string>();
  }
  if (has_file)
  {
    obs.file = resolveConfigPath(cfg_dir, obs.file).string();
  }

  if (node.contains("components"))
  {
    obs.comps = parseIndexList(node.at("components"),
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

BoundarySelector parseSelector(const json&   node,
                               const string& name)
{
  BoundarySelector sel;
  if (node.is_number_integer())
  {
    sel.physical = node.get<Index>();
    return sel;
  }
  if (node.is_string())
  {
    sel.name = node.get<string>();
    return sel;
  }
  if (!node.is_object())
  {
    throw runtime_error(name + " must be a tag, name, or object");
  }

  if (node.contains("tag"))
  {
    sel.physical = node.at("tag").get<Index>();
  }
  assign(node, "name", sel.name);
  return sel;
}

void parseMesh(const json&             node,
               const filesystem::path& cfg_dir,
               MeshParams&             mesh)
{
  if (!node.is_object())
  {
    throw runtime_error("Config mesh must be an object");
  }

  assign(node, "file", mesh.file);

  if (!mesh.file.empty())
  {
    mesh.file = resolveConfigPath(cfg_dir, mesh.file).string();
  }
}

void parseFluid(const json&  node,
                FluidConfig& fluid)
{
  if (!node.is_object())
  {
    throw runtime_error("Config fluid must be an object");
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

void parseTarget(const json&   node,
                 TargetParams& target)
{
  if (!node.is_object())
  {
    throw runtime_error("Boundary velocity must be an object");
  }

  assign(node, "type", target.type);
  assign(node, "quantity", target.qty);
  assign(node, "bulk_speed", target.bulk_speed);
  if (node.contains("mean_velocity"))
  {
    target.bulk_speed = node.at("mean_velocity").get<Real>();
    target.qty        = "mean_velocity";
  }
  else if (node.contains("max_velocity"))
  {
    target.bulk_speed = node.at("max_velocity").get<Real>();
    target.qty        = "max_velocity";
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
  assign(node, "period", target.per);
  assign(node, "radius", target.rad);
  if (node.contains("center"))
  {
    target.cen = parseVector3(node.at("center"), "simulation.center");
  }
  if (node.contains("normal"))
  {
    target.nrm = parseVector3(node.at("normal"), "simulation.normal");
  }
}

void parseTime(const json&   node,
               TargetParams& target)
{
  if (!node.is_object())
  {
    throw runtime_error("Boundary time profile must be an object");
  }

  string type = "uniform";
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
    throw runtime_error(
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
    target.qty        = "mean_velocity";
  }
  else if (node.contains("max_velocity"))
  {
    target.bulk_speed = node.at("max_velocity").get<Real>();
    target.qty        = "max_velocity";
  }

  if (node.contains("amplitude"))
  {
    target.pulse_amplitude = node.at("amplitude").get<Real>();
  }
  else if (node.contains("pulse_amplitude"))
  {
    target.pulse_amplitude = node.at("pulse_amplitude").get<Real>();
  }
  assign(node, "period", target.per);
}

void parseSpace(const json&   node,
                TargetParams& target)
{
  if (!node.is_object())
  {
    throw runtime_error("Boundary space profile must be an object");
  }

  string type = "poiseuille";
  assign(node, "type", type);
  if (type != "poiseuille")
  {
    throw runtime_error("Boundary space.type must be 'poiseuille'");
  }

  target.type = "poiseuille_pulse";
  assign(node, "quantity", target.qty);
  assign(node, "radius", target.rad);
  if (node.contains("center"))
  {
    target.cen =
        parseVector3(node.at("center"), "simulation.bcs.space.center");
  }
  if (node.contains("normal"))
  {
    target.nrm =
        parseVector3(node.at("normal"), "simulation.bcs.space.normal");
  }
}

TargetParams parseVelocity(const json& node)
{
  TargetParams velocity;
  if (!node.contains("velocity") || !node.at("velocity").is_object())
  {
    throw runtime_error("Boundary velocity must be an object");
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

optional<Real> optionalReal(const json& node,
                            const char* key)
{
  if (!node.contains(key))
  {
    return nullopt;
  }
  if (!node.at(key).is_number())
  {
    throw runtime_error(string("Boundary value ") + key
                        + " must be a number");
  }
  return node.at(key).get<Real>();
}

BCsParams parseBc(const json& node)
{
  if (!node.is_object())
  {
    throw runtime_error("Forward boundary condition must be an object");
  }

  BCsParams cond;
  assign(node, "name", cond.name);
  if (node.contains("tag"))
  {
    cond.physical = node.at("tag").get<Index>();
  }
  else
  {
    throw runtime_error("Each boundary condition needs tag");
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
    throw runtime_error(
        "Forward boundary condition needs at least one of ux, uy, uz, p, or velocity");
  }
  return cond;
}

Vector<BCsParams> parseBcList(const json&   node,
                              const string& name)
{
  if (!node.is_array())
  {
    throw runtime_error("Config " + name + " must be an array");
  }

  Vector<BCsParams> out;
  for (const auto& item : node)
  {
    out.push_back(parseBc(item));
  }
  return out;
}

void parseForward(const json&             node,
                  const filesystem::path& cfg_dir,
                  ForwardParams&          forward)
{
  if (!node.is_object())
  {
    throw runtime_error("Config simulation must be an object");
  }

  if (node.contains("mesh"))
  {
    parseMesh(node.at("mesh"), cfg_dir, forward.mesh);
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

void parseOpt(const json&      node,
              OptimizerParams& opt)
{
  if (!node.is_object())
  {
    throw runtime_error("Config optimizer must be an object");
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
      throw runtime_error("Config optimizer.scale must be an object");
    }
    assign(scale, "init_vel", opt.scale.init_vel);
    assign(scale, "boundary", opt.scale.bdry);
  }
}

void parseReg(const json&           node,
              RegularizationParams& reg)
{
  if (!node.is_object())
  {
    throw runtime_error("Config inverse.reg must be an object");
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

void parseObjective(const json&    node,
                    InverseParams& inverse)
{
  if (!node.is_object())
  {
    throw runtime_error("Config objective must be an object");
  }
  assign(node, "alpha", inverse.alpha);
  if (node.contains("reg"))
  {
    parseReg(node.at("reg"), inverse.reg);
  }
}

void parseInitialVelocity(const json&            node,
                          InitialVelocityParams& init_vel)
{
  if (!node.is_object())
  {
    throw runtime_error(
        "Config controls.init_vel must be an object");
  }

  assign(node, "enabled", init_vel.enabled);
  if (!node.contains("bounds"))
  {
    return;
  }

  const auto& bnds = node.at("bounds");
  if (!bnds.is_object())
  {
    throw runtime_error(
        "Config controls.init_vel.bounds must be an object");
  }
  if (bnds.contains("min"))
  {
    init_vel.lower = bnds.at("min").get<Real>();
  }
  if (bnds.contains("max"))
  {
    init_vel.upper = bnds.at("max").get<Real>();
  }
}

void parseControlBounds(const json&   node,
                        BoundsParams& bnds,
                        const string& name)
{
  if (!node.is_object())
  {
    throw runtime_error("Config " + name + " must be an object");
  }
  assign(node, "min", bnds.min);
  if (node.contains("max"))
  {
    bnds.max = node.at("max").get<Real>();
  }
  if (node.contains("normal"))
  {
    bnds.nrm = parseVector3(node.at("normal"), name + ".normal");
  }
}

void parseControls(const json&    node,
                   InverseParams& inverse)
{
  if (!node.is_object())
  {
    throw runtime_error("Config controls must be an object");
  }

  if (node.contains("boundary"))
  {
    const auto& bdry = node.at("boundary");
    inverse.ctr      = parseSelector(bdry, "controls.boundary");
    if (bdry.is_object() && bdry.contains("bounds"))
    {
      parseControlBounds(bdry.at("bounds"),
                         inverse.bnds,
                         "controls.boundary.bounds");
    }
  }
  if (node.contains("init_vel"))
  {
    inverse.init_vel.enabled = true;
    parseInitialVelocity(node.at("init_vel"), inverse.init_vel);
  }
}

void parseInitialGuess(const json&         node,
                       InitialGuessParams& initial_guess)
{
  if (!node.is_object())
  {
    throw runtime_error("Config initial_guess must be an object");
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

void parseInv(const json&             node,
              const filesystem::path& cfg_dir,
              InverseParams&          inverse)
{
  if (!node.is_object())
  {
    throw runtime_error("Config inverse must be an object");
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
    parseObs(node.at("obs"), cfg_dir, inverse.obs);
  }
}

void parseOutput(const json&   node,
                 OutputParams& output)
{
  if (!node.is_object())
  {
    throw runtime_error("Config output must be an object");
  }

  assign(node, "basename", output.base);
}

void parseSolver(const json&   node,
                 SolverParams& solver)
{
  if (!node.is_object())
  {
    throw runtime_error("Config solver must be an object");
  }
  assign(node, "type", solver.type);
  assign(node, "backend", solver.backend);
  assign(node, "method", solver.method);
}

bool hasSelector(const BoundarySelector& sel)
{
  return sel.physical > 0 || !sel.name.empty();
}

bool sameTimeGrid(const TimeParams& lhs,
                  const TimeParams& rhs)
{
  const Real scale = max({Real(1.0), abs(lhs.dt), abs(rhs.dt)});
  const Real tol   = 64.0 * numeric_limits<Real>::epsilon() * scale;
  return lhs.steps == rhs.steps && abs(lhs.dt - rhs.dt) <= tol;
}

void validateSelector(const BoundarySelector& sel,
                      const string&           name)
{
  if (!hasSelector(sel))
  {
    throw runtime_error(name + " requires a physical tag or name");
  }
}

void validate(const Params& prm)
{
  if (prm.fwd.mesh.file.empty())
  {
    throw runtime_error("simulation.mesh.file is required");
  }
  if (prm.fwd.time.steps <= 0 || !isfinite(prm.fwd.time.dt)
      || prm.fwd.time.dt <= 0.0)
  {
    throw runtime_error("Time steps and dt must be positive");
  }
  if (prm.fwd.fluid.rho <= 0.0)
  {
    throw runtime_error("Fluid rho must be positive");
  }
  if (prm.fwd.fluid.mu && *prm.fwd.fluid.mu <= 0.0)
  {
    throw runtime_error("Fluid mu must be positive");
  }
  if (prm.fwd.fluid.Re && *prm.fwd.fluid.Re <= 0.0)
  {
    throw runtime_error("Fluid reynolds must be positive");
  }
  if (!prm.fwd.fluid.mu && !prm.fwd.fluid.Re)
  {
    throw runtime_error("Fluid requires either mu or reynolds");
  }
  if (prm.fwd.bcs.empty())
  {
    throw runtime_error("simulation.bcs must contain at least one boundary");
  }
  for (const auto& bc : prm.fwd.bcs)
  {
    if (bc.physical <= 0)
    {
      throw runtime_error("simulation.bcs physical tag must be positive");
    }
    if (bc.type != "dirichlet")
    {
      throw runtime_error("Only dirichlet simulation.bcs are supported");
    }
    if (bc.vel)
    {
      const auto& velocity = *bc.vel;
      if (velocity.type != "poiseuille_pulse")
      {
        throw runtime_error(
            "simulation.bcs velocity.type must be 'poiseuille_pulse'");
      }
      if (velocity.bulk_speed <= 0.0 || velocity.per <= 0.0
          || velocity.rad <= 0.0)
      {
        throw runtime_error(
            "simulation.bcs velocity bulk_speed, period, and radius must be positive");
      }
      if (velocity.qty != "mean_velocity"
          && velocity.qty != "bulk_speed"
          && velocity.qty != "max_velocity")
      {
        throw runtime_error(
            "simulation.bcs velocity quantity must be 'mean_velocity', 'bulk_speed', or 'max_velocity'");
      }
      Real normal_norm2 = 0.0;
      for (Real value : velocity.nrm)
      {
        normal_norm2 += value * value;
      }
      if (normal_norm2 <= 1.0e-28)
      {
        throw runtime_error("simulation.bcs velocity.normal must be nonzero");
      }
    }
  }
  validateSelector(prm.inv.ctr, "inverse.ctr");
  const BCsParams& ctr_bc = controlBoundary(prm);
  pressureGauge(prm);
  if (fixedVelocityBcs(prm).empty())
  {
    throw runtime_error(
        "simulation.bcs must contain at least one fixed velocity boundary");
  }
  if (prm.fwd.fluid.Re && !ctr_bc.vel)
  {
    throw runtime_error(
        "simulation.bcs control velocity profile is required when fluid.Re is used");
  }
  if (prm.fwd.solver.type != "auto"
      && prm.fwd.solver.type != "resolve"
      && prm.fwd.solver.type != "petsc")
  {
    throw runtime_error(
        "simulation.solver.type must be 'auto', 'resolve', or 'petsc'");
  }
  if (prm.fwd.solver.backend != "cpu"
      && prm.fwd.solver.backend != "cuda")
  {
    throw runtime_error(
        "simulation.solver.backend must be 'cpu' or 'cuda'");
  }
  if (prm.fwd.solver.method != "direct"
      && prm.fwd.solver.method != "iterative")
  {
    throw runtime_error(
        "simulation.solver.method must be 'direct' or 'iterative'");
  }
  if (prm.inv.alpha < 0.0
      || prm.inv.reg.beta1 < 0.0
      || prm.inv.reg.beta2 < 0.0
      || prm.inv.reg.beta3 < 0.0
      || prm.inv.reg.beta4 < 0.0)
  {
    throw runtime_error(
        "inverse alpha and reg values must be nonnegative");
  }
  if (prm.inv.init_vel.lower
      && prm.inv.init_vel.upper
      && *prm.inv.init_vel.lower
             > *prm.inv.init_vel.upper)
  {
    throw runtime_error(
        "inverse.init_vel lower must not exceed upper");
  }
  if (prm.inv.opt.max_iterations < 0
      || prm.inv.opt.abs_tol < 0.0
      || prm.inv.opt.rel_tol < 0.0
      || prm.inv.opt.step_tol < 0.0)
  {
    throw runtime_error(
        "inverse opt tolerances and max_iterations must be nonnegative");
  }
  if (!isfinite(prm.inv.opt.scale.init_vel)
      || prm.inv.opt.scale.init_vel <= 0.0
      || !isfinite(prm.inv.opt.scale.bdry)
      || prm.inv.opt.scale.bdry <= 0.0)
  {
    throw runtime_error("optimizer.scale values must be positive");
  }
  if (prm.inv.opt.type != "lmvm")
  {
    throw runtime_error("inverse.opt.type must be 'lmvm'");
  }
  if (prm.inv.bnds.max && *prm.inv.bnds.max < prm.inv.bnds.min)
  {
    throw runtime_error(
        "inverse.bounds.max must be greater than min");
  }
  if (!prm.inv.bnds.max && !ctr_bc.vel)
  {
    throw runtime_error(
        "inverse.bounds.max is required when the control bcs has no velocity profile");
  }
  if (prm.inv.bnds.max_scale <= 0.0)
  {
    throw runtime_error("inverse.bounds.max_scale must be positive");
  }
  Real bounds_normal_norm2 = 0.0;
  for (Real value : prm.inv.bnds.nrm)
  {
    bounds_normal_norm2 += value * value;
  }
  if (bounds_normal_norm2 <= 1.0e-28)
  {
    throw runtime_error("inverse.bounds.normal must be nonzero");
  }
  if (prm.inv.obs.file.empty())
  {
    throw runtime_error("inverse.obs.file is required");
  }
}

Real interpolatedControlParam(const DirichletControl&            ctr,
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
    throw runtime_error("control interpolation step is out of range");
  }
  if (ctr.numDofs() <= 0
      || prm.size() % ctr.numDofs() != 0)
  {
    throw runtime_error("control parameter layout is invalid");
  }

  const LinearInterpolation&  interp = stencils[step];
  Real                        value  = 0.0;
  BlockVectorView<const Real> levels(
      prm.data(), prm.size() / ctr.numDofs(), ctr.numDofs());
  interp.forEachWeight(
      [&](Index level, Real wt)
      {
        if (level < 0 || level >= levels.blocks())
        {
          throw runtime_error("control interpolation level is out of range");
        }
        value += wt * levels(level, i);
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
  resizeOrZero(ux, nodes);
  resizeOrZero(uy, nodes);
  resizeOrZero(uz, nodes);
  resizeOrZero(p, nodes);

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
                  const DirichletControl&            ctr,
                  const Vector<Real>&                prm,
                  const Vector<LinearInterpolation>& stencils,
                  Index                              step,
                  Vector<Real>&                      ux,
                  Vector<Real>&                      uy,
                  Vector<Real>&                      uz)
{
  const Index nodes = space.mesh().numNodes();
  resizeOrZero(ux, nodes);
  resizeOrZero(uy, nodes);
  resizeOrZero(uz, nodes);
  if (step < 0)
  {
    return;
  }

  for (Index i = 0; i < ctr.numDofs(); ++i)
  {
    assignControlDof(space,
                     ctr.stateDof(i),
                     interpolatedControlParam(ctr, prm, stencils, step, i),
                     ux,
                     uy,
                     uz);
  }
}

void ensureDir(const string& base)
{
  const filesystem::path path(base);
  const filesystem::path dir = path.parent_path();
  if (!dir.empty())
  {
    filesystem::create_directories(dir);
  }
}

} // namespace

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
  if (!root.is_object())
  {
    throw runtime_error("Config root must be an object");
  }

  parseForward(root, cfg_dir, prm.fwd);
  parseInv(root, cfg_dir, prm.inv);

  if (prm.fwd.bcs.empty() && !prm.inv.initial_guess.bcs.empty())
  {
    prm.fwd.bcs = prm.inv.initial_guess.bcs;
  }
  if (prm.inv.initial_guess.has_time
      && !sameTimeGrid(prm.fwd.time, prm.inv.initial_guess.time))
  {
    throw runtime_error(
        "initial_guess.time currently must match the simulation time grid");
  }

  validate(prm);
  return prm;
}

void writeResultViz(const Mesh&                        mesh,
                    const MixedFESpace&                space,
                    const DirichletControl&            ctr,
                    const TimeTrajectory&              tr,
                    const Vector<Real>&                prm,
                    const Vector<LinearInterpolation>& stencils,
                    Real                               dt,
                    const VizOptions&                  opts,
                    Real                               time_offset)
{
  if (stencils.size() != tr.numSteps())
  {
    throw runtime_error(
        "writeResultViz control time stencil size mismatch");
  }
  if (ctr.numDofs() > 0
      && (prm.empty() || prm.size() % ctr.numDofs() != 0))
  {
    throw runtime_error("writeResultViz control parameter size mismatch");
  }
  const Index nctr =
      ctr.numDofs() == 0 ? 0 : prm.size() / ctr.numDofs();
  for (const LinearInterpolation& stencil : stencils)
  {
    if (!stencil.isValid() || stencil.upper >= nctr)
    {
      throw runtime_error(
          "writeResultViz control stencil is outside parameter levels");
    }
  }

  ensureDir(opts.base);

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
    controlField(space, ctr, prm, stencils, step, ctr_x, ctr_y, ctr_z);
    out.addNodalVectorField("control", ctr_x, ctr_y, ctr_z);
  }

  out.write(opts.base);
}

BoundarySelector bcSelector(const BCsParams& bc)
{
  return BoundarySelector{bc.physical, bc.name};
}

namespace
{

bool matches(const BoundarySelector& sel,
             const BCsParams&        bc)
{
  if (!sel.name.empty())
  {
    return bc.name == sel.name;
  }
  return bc.physical == sel.physical;
}

bool hasFixedVel(const BCsParams& bc)
{
  return bc.ux || bc.uy || bc.uz || bc.vel;
}

Real reLength(const TargetParams& target)
{
  return 2.0 * target.rad;
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
  throw runtime_error("simulation.bcs must contain a pressure boundary");
}

Vector<BoundarySelector> fixedVelocityBcs(const Params& prm)
{
  Vector<BoundarySelector> selectors;
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
  throw runtime_error("inverse.ctr did not match any simulation.bcs");
}

const TargetParams& controlTarget(const Params& prm)
{
  const BCsParams& bc = controlBoundary(prm);
  if (!bc.vel)
  {
    throw runtime_error(
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
