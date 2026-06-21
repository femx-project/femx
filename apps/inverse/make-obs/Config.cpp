#include "Config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace femx::make_obs
{
namespace
{

using namespace femx::navier_var_new;

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

std::filesystem::path resolveConfigPath(
    const std::filesystem::path& config_dir,
    const std::string&           path)
{
  const std::filesystem::path candidate(path);
  if (candidate.is_absolute() || config_dir.empty())
  {
    return candidate;
  }
  return (config_dir / candidate).lexically_normal();
}

std::string stripKnownOutputExtension(std::string path)
{
  const auto strip = [&path](const std::string& ext)
  {
    if (path.size() >= ext.size()
        && path.compare(path.size() - ext.size(), ext.size(), ext) == 0)
    {
      path.resize(path.size() - ext.size());
      return true;
    }
    return false;
  };

  strip(".txt") || strip(".dat") || strip(".obs") || strip(".vti")
      || strip(".pvd");
  return path;
}

Index stepsForEndTime(Real end_time,
                      Real dt)
{
  if (!std::isfinite(end_time) || end_time <= 0.0)
  {
    throw std::runtime_error("forward.time.end_time must be positive");
  }
  if (!std::isfinite(dt) || dt <= 0.0)
  {
    throw std::runtime_error("forward.time.dt must be positive");
  }

  const Real scaled = end_time / dt;
  if (!std::isfinite(scaled) || scaled <= 0.0
      || scaled > static_cast<Real>(std::numeric_limits<Index>::max()))
  {
    throw std::runtime_error("forward.time.end_time / dt is out of range");
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
    throw std::runtime_error("Config forward.time must be an object");
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
          "forward.time steps and end_time disagree for the configured dt");
    }
    time.steps = derived_steps;
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

void parseObsGrid(const nlohmann::json&    node,
                  ObservationParams::Grid& grid)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config obs.grid must be an object");
  }

  if (node.contains("counts"))
  {
    grid.counts = parseIndex3(node.at("counts"), "obs.grid.counts");
  }
  else if (node.contains("shape"))
  {
    grid.counts = parseIndex3(node.at("shape"), "obs.grid.shape");
  }

  if (node.contains("bounds"))
  {
    const auto& bounds = node.at("bounds");
    if (!bounds.is_array() || bounds.size() != 2)
    {
      throw std::runtime_error(
          "Config obs.grid.bounds must contain lower and upper points");
    }
    grid.lower = parseVector3(bounds.at(0), "obs.grid.bounds[0]");
    grid.upper = parseVector3(bounds.at(1), "obs.grid.bounds[1]");
  }
  if (node.contains("lower"))
  {
    grid.lower = parseVector3(node.at("lower"), "obs.grid.lower");
  }
  else if (node.contains("min"))
  {
    grid.lower = parseVector3(node.at("min"), "obs.grid.min");
  }
  if (node.contains("upper"))
  {
    grid.upper = parseVector3(node.at("upper"), "obs.grid.upper");
  }
  else if (node.contains("max"))
  {
    grid.upper = parseVector3(node.at("max"), "obs.grid.max");
  }

  if (node.contains("origin"))
  {
    grid.origin      = parseVector3(node.at("origin"), "obs.grid.origin");
    grid.use_spacing = true;
  }
  if (node.contains("spacing"))
  {
    grid.spacing     = parseVector3(node.at("spacing"), "obs.grid.spacing");
    grid.use_spacing = true;
  }
}

void parseObs(const nlohmann::json& node,
              ObservationParams&    obs)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config obs must be an object");
  }

  if (node.contains("type"))
  {
    const std::string type = node.at("type").get<std::string>();
    if (type != "grid")
    {
      throw std::runtime_error("Config obs.type must be 'grid'");
    }
    obs.type = "grid";
  }
  if (node.contains("components"))
  {
    obs.components = parseIndexList(node.at("components"),
                                    "obs.components");
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

void parseMesh(const nlohmann::json&        node,
               const std::filesystem::path& config_dir,
               MeshParams&                  mesh)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config forward.mesh must be an object");
  }

  assign(node, "file", mesh.file);
  if (!mesh.file.empty())
  {
    mesh.file = resolveConfigPath(config_dir, mesh.file).string();
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
    target.center = parseVector3(node.at("center"), "forward.bcs.center");
  }
  if (node.contains("normal"))
  {
    target.normal = parseVector3(node.at("normal"), "forward.bcs.normal");
  }
}

void parseTimeProfile(const nlohmann::json& node,
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

void parseSpaceProfile(const nlohmann::json& node,
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
    target.center = parseVector3(node.at("center"), "forward.bcs.space.center");
  }
  if (node.contains("normal"))
  {
    target.normal = parseVector3(node.at("normal"), "forward.bcs.space.normal");
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
    parseTimeProfile(node.at("time"), velocity);
  }
  if (node.contains("space"))
  {
    parseSpaceProfile(node.at("space"), velocity);
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
  if (node.contains("velocity") || node.contains("time")
      || node.contains("space"))
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

void parseBcs(const nlohmann::json& node,
              ForwardParams&        forward)
{
  if (!node.is_array())
  {
    throw std::runtime_error("Config forward.bcs must be an array");
  }

  forward.bcs.clear();
  for (const auto& item : node)
  {
    forward.bcs.push_back(parseBc(item));
  }
}

void parseFluid(const nlohmann::json& node,
                FluidConfig&          fluid)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config forward.fluid must be an object");
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

void parseOutputViz(const nlohmann::json&         node,
                    const std::filesystem::path&  config_dir,
                    navier_var_new::OutputParams& output)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config forward.output must be an object");
  }

  const bool has_basename = node.contains("basename");
  assign(node, "basename", output.basename);
  if (has_basename && !output.basename.empty())
  {
    output.basename = resolveConfigPath(config_dir, output.basename).string();
  }
}

void parseSolver(const nlohmann::json& node,
                 SolverParams&         solver)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config forward.solver must be an object");
  }
  assign(node, "type", solver.type);
  assign(node, "backend", solver.backend);
}

void parseForward(const nlohmann::json&        node,
                  const std::filesystem::path& config_dir,
                  ForwardParams&               forward)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config forward must be an object");
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
    parseOutputViz(node.at("output"), config_dir, forward.output);
  }
  if (node.contains("solver"))
  {
    parseSolver(node.at("solver"), forward.solver);
  }
}

void resolveOutputPaths(const std::filesystem::path& config_dir,
                        OutputParams&                output)
{
  if (!output.file.empty())
  {
    output.file = resolveConfigPath(config_dir, output.file).string();
  }
  if (output.write_vti && output.vti_basename.empty())
  {
    output.vti_basename = stripKnownOutputExtension(output.file);
  }
  if (!output.vti_basename.empty())
  {
    output.vti_basename =
        resolveConfigPath(config_dir, output.vti_basename).string();
    output.write_vti = true;
  }
  const bool has_explicit_reference = !output.reference_basename.empty();
  if (output.write_reference && output.reference_basename.empty())
  {
    output.reference_basename =
        stripKnownOutputExtension(output.file) + "-reference";
  }
  if (has_explicit_reference)
  {
    output.reference_basename =
        resolveConfigPath(config_dir, output.reference_basename).string();
  }
}

void parseOutput(const nlohmann::json& node,
                 OutputParams&         output)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config make_obs.output must be an object");
  }

  assign(node, "file", output.file);
  assign(node, "path", output.file);
  assign(node, "write_vti", output.write_vti);
  assign(node, "vti", output.vti_basename);
  assign(node, "vti_file", output.vti_basename);
  assign(node, "vti_path", output.vti_basename);
  assign(node, "vti_basename", output.vti_basename);
  assign(node, "write_reference", output.write_reference);
  assign(node, "reference", output.reference_basename);
  assign(node, "reference_file", output.reference_basename);
  assign(node, "reference_path", output.reference_basename);
  assign(node, "reference_basename", output.reference_basename);
}

bool hasOutputKeys(const nlohmann::json& node)
{
  return node.contains("file") || node.contains("path")
         || node.contains("write_vti") || node.contains("vti")
         || node.contains("vti_file") || node.contains("vti_path")
         || node.contains("vti_basename")
         || node.contains("write_reference") || node.contains("reference")
         || node.contains("reference_file") || node.contains("reference_path")
         || node.contains("reference_basename");
}

void resolveInputPaths(const std::filesystem::path& config_dir,
                       InputParams&                 input)
{
  if (!input.trajectory.empty())
  {
    input.trajectory =
        resolveConfigPath(config_dir, input.trajectory).string();
  }
}

void parseInput(const nlohmann::json&        node,
                const std::filesystem::path& config_dir,
                InputParams&                 input)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config make_obs.input must be an object");
  }
  assign(node, "trajectory", input.trajectory);
  assign(node, "trajectory_file", input.trajectory);
  assign(node, "trajectory_path", input.trajectory);
  assign(node, "velocity_field", input.velocity_field);
  (void) config_dir;
}

void parseNoise(const nlohmann::json& node,
                NoiseParams&          noise)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config make_obs.noise must be an object");
  }

  const bool has_enabled = node.contains("enabled");
  assign(node, "enabled", noise.enabled);
  if (node.contains("snr"))
  {
    noise.snr = node.at("snr").get<Real>();
    if (!has_enabled)
    {
      noise.enabled = true;
    }
  }
  else if (node.contains("ratio"))
  {
    noise.snr = node.at("ratio").get<Real>();
    if (!has_enabled)
    {
      noise.enabled = true;
    }
  }
  else if (node.contains("snr_ratio"))
  {
    noise.snr = node.at("snr_ratio").get<Real>();
    if (!has_enabled)
    {
      noise.enabled = true;
    }
  }

  if (node.contains("snr_db"))
  {
    noise.snr_db = node.at("snr_db").get<Real>();
    if (!has_enabled)
    {
      noise.enabled = true;
    }
  }
  else if (node.contains("snr_dB"))
  {
    noise.snr_db = node.at("snr_dB").get<Real>();
    if (!has_enabled)
    {
      noise.enabled = true;
    }
  }
}

void assignOptionalReal(const nlohmann::json& node,
                        const char*           key,
                        std::optional<Real>&  value)
{
  if (node.contains(key))
  {
    value = node.at(key).get<Real>();
  }
}

void assignOptionalIndex(const nlohmann::json& node,
                         const char*           key,
                         std::optional<Index>& value)
{
  if (node.contains(key))
  {
    value = node.at(key).get<Index>();
  }
}

void parseTimeSample(const nlohmann::json& node,
                     TimeSampleParams&     time)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config make_obs.time must be an object");
  }

  assignOptionalReal(node, "start_time", time.start_time);
  assignOptionalReal(node, "begin_time", time.start_time);
  assignOptionalReal(node, "end_time", time.end_time);
  assignOptionalReal(node, "finish_time", time.end_time);
  assignOptionalIndex(node, "start_level", time.start_level);
  assignOptionalIndex(node, "begin_level", time.start_level);
  assignOptionalIndex(node, "end_level", time.end_level);
  assignOptionalIndex(node, "finish_level", time.end_level);
  assignOptionalIndex(node, "num_points", time.num_points);
  assignOptionalIndex(node, "num_levels", time.num_points);
  assignOptionalIndex(node, "levels", time.num_points);
  assignOptionalIndex(node, "count", time.num_points);
}

bool hasObsKeys(const nlohmann::json& node)
{
  return node.contains("type") || node.contains("components")
         || node.contains("grid") || node.contains("counts");
}

bool isObservationCaseKey(const std::string& key)
{
  if (key.size() <= 3 || key.rfind("obs", 0) != 0)
  {
    return false;
  }
  for (std::size_t i = 3; i < key.size(); ++i)
  {
    if (key[i] < '0' || key[i] > '9')
    {
      return false;
    }
  }
  return true;
}

void makeCaseOutputUnique(ObservationCase& item)
{
  const std::string root = stripKnownOutputExtension(item.output.file);
  if (root.empty())
  {
    return;
  }

  item.output.file = root + "-" + item.name + ".txt";
  if (item.output.write_vti || !item.output.vti_basename.empty())
  {
    item.output.write_vti    = true;
    item.output.vti_basename = {};
  }
  if (item.output.write_reference
      || !item.output.reference_basename.empty())
  {
    item.output.write_reference    = true;
    item.output.reference_basename = {};
  }
}

ObservationCase parseObservationCase(
    const nlohmann::json&        node,
    const std::filesystem::path& config_dir,
    const Params&                defaults,
    std::string                  name)
{
  if (!node.is_object())
  {
    throw std::runtime_error(
        "Config make_obs.observations." + name + " must be an object");
  }

  ObservationCase item;
  item.name   = std::move(name);
  item.obs    = defaults.obs;
  item.output = defaults.output;
  item.noise  = defaults.noise;
  item.time   = defaults.time;

  assign(node, "name", item.name);

  if (node.contains("obs"))
  {
    parseObs(node.at("obs"), item.obs);
  }
  if (hasObsKeys(node))
  {
    parseObs(node, item.obs);
  }

  bool has_explicit_output = false;
  if (node.contains("output"))
  {
    parseOutput(node.at("output"), item.output);
    has_explicit_output = true;
  }
  if (hasOutputKeys(node))
  {
    parseOutput(node, item.output);
    has_explicit_output = true;
  }
  if (!has_explicit_output)
  {
    makeCaseOutputUnique(item);
  }

  if (node.contains("noise"))
  {
    parseNoise(node.at("noise"), item.noise);
  }
  if (node.contains("snr") || node.contains("snr_db")
      || node.contains("snr_dB") || node.contains("ratio")
      || node.contains("snr_ratio"))
  {
    parseNoise(node, item.noise);
  }
  if (node.contains("time"))
  {
    parseTimeSample(node.at("time"), item.time);
  }

  (void) config_dir;
  return item;
}

void parseObservationCases(const nlohmann::json&        node,
                           const std::filesystem::path& config_dir,
                           Params&                      prm)
{
  if (node.is_object())
  {
    for (auto it = node.begin(); it != node.end(); ++it)
    {
      prm.observations.push_back(
          parseObservationCase(it.value(), config_dir, prm, it.key()));
    }
    return;
  }

  if (!node.is_array())
  {
    throw std::runtime_error(
        "Config make_obs.observations must be an object or array");
  }
  for (std::size_t i = 0; i < node.size(); ++i)
  {
    prm.observations.push_back(parseObservationCase(
        node.at(i), config_dir, prm, "obs" + std::to_string(i + 1)));
  }
}

void parseTopLevelObservationCases(const nlohmann::json&        root,
                                   const std::filesystem::path& config_dir,
                                   Params&                      prm)
{
  for (auto it = root.begin(); it != root.end(); ++it)
  {
    if (isObservationCaseKey(it.key()))
    {
      prm.observations.push_back(
          parseObservationCase(it.value(), config_dir, prm, it.key()));
    }
  }
}

void parseMakeObs(const nlohmann::json&        node,
                  const std::filesystem::path& config_dir,
                  Params&                      prm)
{
  if (!node.is_object())
  {
    throw std::runtime_error("Config make_obs must be an object");
  }

  if (node.contains("obs"))
  {
    parseObs(node.at("obs"), prm.obs);
  }
  if (node.contains("input"))
  {
    parseInput(node.at("input"), config_dir, prm.input);
  }
  if (node.contains("trajectory") || node.contains("trajectory_file")
      || node.contains("trajectory_path") || node.contains("velocity_field"))
  {
    parseInput(node, config_dir, prm.input);
  }
  if (node.contains("output"))
  {
    parseOutput(node.at("output"), prm.output);
  }
  if (hasOutputKeys(node))
  {
    parseOutput(node, prm.output);
  }
  if (node.contains("noise"))
  {
    parseNoise(node.at("noise"), prm.noise);
  }
  if (node.contains("snr") || node.contains("snr_db")
      || node.contains("snr_dB") || node.contains("ratio")
      || node.contains("snr_ratio"))
  {
    parseNoise(node, prm.noise);
  }
  if (node.contains("time"))
  {
    parseTimeSample(node.at("time"), prm.time);
  }
  if (node.contains("observations"))
  {
    parseObservationCases(node.at("observations"), config_dir, prm);
  }
  else if (node.contains("obs_cases"))
  {
    parseObservationCases(node.at("obs_cases"), config_dir, prm);
  }
  else if (node.contains("cases"))
  {
    parseObservationCases(node.at("cases"), config_dir, prm);
  }
}

bool hasPressureBoundary(const ForwardParams& forward)
{
  for (const auto& bc : forward.bcs)
  {
    if (bc.p)
    {
      return true;
    }
  }
  return false;
}

bool hasTimeSample(const TimeSampleParams& time)
{
  return time.start_time || time.end_time || time.start_level
         || time.end_level || time.num_points;
}

void validateTarget(const TargetParams& target)
{
  if (target.type != "poiseuille_pulse")
  {
    throw std::runtime_error(
        "forward.bcs velocity.type must be 'poiseuille_pulse'");
  }
  if (target.bulk_speed <= 0.0 || target.period <= 0.0
      || target.radius <= 0.0)
  {
    throw std::runtime_error(
        "forward.bcs velocity bulk_speed, period, and radius must be positive");
  }
  if (target.quantity != "mean_velocity"
      && target.quantity != "bulk_speed"
      && target.quantity != "max_velocity")
  {
    throw std::runtime_error(
        "forward.bcs velocity quantity must be 'mean_velocity', 'bulk_speed', or 'max_velocity'");
  }
  Real normal_norm2 = 0.0;
  for (Real value : target.normal)
  {
    normal_norm2 += value * value;
  }
  if (normal_norm2 <= 1.0e-28)
  {
    throw std::runtime_error("forward.bcs velocity.normal must be nonzero");
  }
}

void validateForward(const ForwardParams& forward)
{
  if (forward.mesh.file.empty())
  {
    throw std::runtime_error("forward.mesh.file is required");
  }
  if (forward.time.steps <= 0 || !std::isfinite(forward.time.dt)
      || forward.time.dt <= 0.0)
  {
    throw std::runtime_error("Time steps and dt must be positive");
  }
  if (forward.fluid.rho <= 0.0)
  {
    throw std::runtime_error("Fluid rho must be positive");
  }
  if (forward.fluid.mu && *forward.fluid.mu <= 0.0)
  {
    throw std::runtime_error("Fluid mu must be positive");
  }
  if (forward.fluid.Re && *forward.fluid.Re <= 0.0)
  {
    throw std::runtime_error("Fluid reynolds must be positive");
  }
  if (!forward.fluid.mu && !forward.fluid.Re)
  {
    throw std::runtime_error("Fluid requires either mu or reynolds");
  }
  if (forward.bcs.empty())
  {
    throw std::runtime_error("forward.bcs must contain at least one boundary");
  }
  if (!hasPressureBoundary(forward))
  {
    throw std::runtime_error("forward.bcs must contain a pressure boundary");
  }
  for (const auto& bc : forward.bcs)
  {
    if (bc.physical <= 0)
    {
      throw std::runtime_error("forward.bcs physical tag must be positive");
    }
    if (bc.type != "dirichlet")
    {
      throw std::runtime_error("Only dirichlet forward.bcs are supported");
    }
    if (bc.vel)
    {
      validateTarget(*bc.vel);
    }
  }
  if (forward.solver.type != "auto" && forward.solver.type != "resolve"
      && forward.solver.type != "petsc")
  {
    throw std::runtime_error(
        "forward.solver.type must be 'auto', 'resolve', or 'petsc'");
  }
  if (forward.solver.backend != "cpu"
      && forward.solver.backend != "cuda")
  {
    throw std::runtime_error("forward.solver.backend must be 'cpu' or 'cuda'");
  }
}

void validate(const Params& prm)
{
  if (prm.input.velocity_field.empty())
  {
    throw std::runtime_error("make_obs.input.velocity_field must not be empty");
  }

  const auto validate_output = [](const OutputParams& output,
                                  const std::string&  prefix)
  {
    if (output.file.empty())
    {
      throw std::runtime_error(prefix + ".output.file is required");
    }
  };
  const auto validate_noise = [](const NoiseParams& noise,
                                 const std::string& prefix)
  {
    if (noise.snr && noise.snr_db)
    {
      throw std::runtime_error(
          prefix + ".noise accepts either snr or snr_db, not both");
    }
    if (noise.enabled)
    {
      if (!noise.snr && !noise.snr_db)
      {
        throw std::runtime_error(
            prefix + ".noise.enabled requires snr or snr_db");
      }
      if (noise.snr
          && (!std::isfinite(*noise.snr) || *noise.snr <= 0.0))
      {
        throw std::runtime_error(prefix + ".noise.snr must be positive");
      }
      if (noise.snr_db && !std::isfinite(*noise.snr_db))
      {
        throw std::runtime_error(prefix + ".noise.snr_db must be finite");
      }
    }
  };
  const auto validate_time = [](const TimeSampleParams& time,
                                const std::string&      prefix)
  {
    if (!hasTimeSample(time))
    {
      return;
    }
    if (time.start_time && time.start_level)
    {
      throw std::runtime_error(
          prefix + ".time accepts either start_time or start_level, not both");
    }
    if (time.end_time && time.end_level)
    {
      throw std::runtime_error(
          prefix + ".time accepts either end_time or end_level, not both");
    }
    if (time.start_time && !std::isfinite(*time.start_time))
    {
      throw std::runtime_error(prefix + ".time.start_time must be finite");
    }
    if (time.end_time && !std::isfinite(*time.end_time))
    {
      throw std::runtime_error(prefix + ".time.end_time must be finite");
    }
    if (time.start_level && *time.start_level < 0)
    {
      throw std::runtime_error(
          prefix + ".time.start_level must be non-negative");
    }
    if (time.end_level && *time.end_level < 0)
    {
      throw std::runtime_error(prefix + ".time.end_level must be non-negative");
    }
    if (time.num_points && *time.num_points <= 0)
    {
      throw std::runtime_error(prefix + ".time.num_points must be positive");
    }
  };

  if (!prm.observations.empty())
  {
    for (const ObservationCase& item : prm.observations)
    {
      if (item.name.empty())
      {
        throw std::runtime_error(
            "make_obs.observations entries must have non-empty names");
      }
      const std::string prefix = "make_obs.observations." + item.name;
      validate_output(item.output, prefix);
      validate_noise(item.noise, prefix);
      validate_time(item.time, prefix);
    }
    return;
  }

  validate_output(prm.output, "make_obs");
  validate_noise(prm.noise, "make_obs");
  validate_time(prm.time, "make_obs");
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

  if (root.contains("obs"))
  {
    parseObs(root.at("obs"), prm.obs);
  }
  if (root.contains("inverse") && root.at("inverse").contains("obs"))
  {
    parseObs(root.at("inverse").at("obs"), prm.obs);
  }
  if (root.contains("make_obs"))
  {
    parseMakeObs(root.at("make_obs"), config_dir, prm);
  }
  if (root.contains("input"))
  {
    parseInput(root.at("input"), config_dir, prm.input);
  }
  if (root.contains("trajectory") || root.contains("trajectory_file")
      || root.contains("trajectory_path"))
  {
    parseInput(root, config_dir, prm.input);
  }
  if (root.contains("output"))
  {
    parseOutput(root.at("output"), prm.output);
  }
  if (hasOutputKeys(root))
  {
    parseOutput(root, prm.output);
  }
  if (root.contains("noise"))
  {
    parseNoise(root.at("noise"), prm.noise);
  }
  if (root.contains("snr") || root.contains("snr_db")
      || root.contains("snr_dB") || root.contains("ratio")
      || root.contains("snr_ratio"))
  {
    parseNoise(root, prm.noise);
  }
  if (root.contains("time"))
  {
    parseTimeSample(root.at("time"), prm.time);
  }
  parseTopLevelObservationCases(root, config_dir, prm);

  resolveInputPaths(config_dir, prm.input);
  resolveOutputPaths(config_dir, prm.output);
  for (ObservationCase& item : prm.observations)
  {
    resolveOutputPaths(config_dir, item.output);
  }
  validate(prm);
  return prm;
}

} // namespace femx::make_obs
