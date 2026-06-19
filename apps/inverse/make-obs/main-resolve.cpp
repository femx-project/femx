#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "Config.hpp"
#include "DirichletControl.hpp"
#include "DirichletControlEquation.hpp"
#include "NavierStokesEquation.hpp"
#include "ObservationVti.hpp"
#include "RunSupport.hpp"
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/common/Workspace.hpp>
#include <femx/eq/TimeMatrixLinearStateSolver.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/GmshReader.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>
#include <femx/system/resolve/ReSolveLinearSolver.hpp>

#ifndef FEMX_MAKE_OBS_APP_NAME
#define FEMX_MAKE_OBS_APP_NAME "make-obs"
#endif

namespace
{

using namespace femx;
using namespace femx::assembly;
using namespace femx::eq;
using namespace femx::inverse;
using namespace femx::make_obs;
using namespace femx::navier_var;
using namespace femx::system;

constexpr Index kQuadOrder = 2;
constexpr auto  kNoiseSeed = 1ULL;

void requireReSolve(const SolverParams& solver)
{
  if (solver.type == "petsc")
  {
    throw std::runtime_error(
        std::string(FEMX_MAKE_OBS_APP_NAME)
        + " requires forward.solver.type='auto' or 'resolve'");
  }
}

WorkspaceType workspaceType(const SolverParams& solver)
{
  if (solver.backend == "cuda")
  {
    return WorkspaceType::Cuda;
  }
  return WorkspaceType::Cpu;
}

ReSolveOptions makeReSolveOptions()
{
  ReSolveOptions opts;
  opts.factor   = "none";
  opts.refactor = "none";
  opts.ir       = "none";
  opts.max_its  = 5000;
  opts.restart  = 200;
  opts.rtol     = 1.0e-8;
  opts.solve    = "fgmres";
  opts.precond  = "ilu0";
  opts.flexible = true;
  return opts;
}

struct AppOptions
{
  std::string          config_file;
  std::optional<Index> steps;
  std::optional<Real>  snr;
  std::optional<Real>  snr_db;
  std::string          output_file;
  bool                 no_noise = false;
  bool                 help     = false;
};

std::string requireValue(int                argc,
                         char**             argv,
                         int&               i,
                         const std::string& key)
{
  if (i + 1 >= argc)
  {
    throw std::runtime_error("Missing value for " + key);
  }
  return std::string(argv[++i]);
}

AppOptions parseAppOptions(int    argc,
                           char** argv)
{
  AppOptions options;
  for (int i = 1; i < argc; ++i)
  {
    const std::string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      options.help = true;
      return options;
    }
    if (key == "--config" || key == "-config")
    {
      options.config_file = requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--steps")
    {
      options.steps = static_cast<Index>(
          std::stoll(requireValue(argc, argv, i, key)));
      if (*options.steps <= 0)
      {
        throw std::runtime_error("--steps must be positive");
      }
      continue;
    }
    if (key == "--output")
    {
      options.output_file = requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--snr")
    {
      options.snr = std::stod(requireValue(argc, argv, i, key));
      continue;
    }
    if (key == "--snr-db")
    {
      options.snr_db = std::stod(requireValue(argc, argv, i, key));
      continue;
    }
    if (key == "--no-noise")
    {
      options.no_noise = true;
      continue;
    }
  }
  return options;
}

void printUsage(std::ostream& out)
{
  out << "Usage: " << FEMX_MAKE_OBS_APP_NAME
      << " --config FILE [--steps N] [--output FILE]"
      << " [--snr R | --snr-db DB]\n";
}

void applyOverrides(const AppOptions& options,
                    make_obs::Params& prm)
{
  if (options.steps)
  {
    prm.forward.time.steps = *options.steps;
  }
  if (!options.output_file.empty())
  {
    prm.output.file = options.output_file;
  }
  if (options.no_noise)
  {
    prm.noise.enabled = false;
    prm.noise.snr.reset();
    prm.noise.snr_db.reset();
  }
  if (options.snr)
  {
    prm.noise.enabled = true;
    prm.noise.snr     = *options.snr;
    prm.noise.snr_db.reset();
  }
  if (options.snr_db)
  {
    prm.noise.enabled = true;
    prm.noise.snr_db  = *options.snr_db;
    prm.noise.snr.reset();
  }
}

class ProgressPrinter
{
public:
  void beginPhase(const std::string& name)
  {
    finishLine();
    std::cout << "  " << name << '\n';
  }

  void timeStep(Index step,
                Index total)
  {
    std::cout << "\r    time step " << std::setw(4) << step
              << " / " << std::setw(4) << total << std::flush;
    line_active_ = true;
    if (step >= total)
    {
      finishLine();
    }
  }

  void finishLine()
  {
    if (line_active_)
    {
      std::cout << '\n';
      line_active_ = false;
    }
  }

private:
  bool line_active_{false};
};

struct BoundaryValues
{
  Vector<Index> dofs;
  Vector<Real>  prm;
};

bool selectorMatches(const BoundarySelector&    selector,
                     const Mesh::BoundaryFacet& facet)
{
  if (!selector.name.empty())
  {
    return facet.physical_name == selector.name;
  }
  return facet.physical_tag == selector.physical;
}

std::set<Index> boundaryNodes(const Mesh&             mesh,
                              const BoundarySelector& selector)
{
  std::set<Index> nodes;
  for (const auto& facet : mesh.boundaryFacets())
  {
    if (!selectorMatches(selector, facet))
    {
      continue;
    }
    for (Index node : facet.node_ids)
    {
      nodes.insert(node);
    }
  }
  if (nodes.empty())
  {
    throw std::runtime_error("make-obs boundary condition matched no nodes");
  }
  return nodes;
}

void addBoundaryValue(std::map<Index, Vector<Real>>& values,
                      Index                          dof,
                      Index                          step,
                      Index                          steps,
                      Real                           value)
{
  auto it = values.find(dof);
  if (it == values.end())
  {
    it = values.emplace(dof, Vector<Real>(steps)).first;
    for (Index i = 0; i < steps; ++i)
    {
      it->second[i] = std::numeric_limits<Real>::quiet_NaN();
    }
  }
  else if (!std::isnan(it->second[step])
           && std::abs(it->second[step] - value) > 1.0e-12)
  {
    throw std::runtime_error(
        "make-obs received conflicting Dirichlet boundary values at dof "
        + std::to_string(dof) + ", step " + std::to_string(step)
        + ": " + std::to_string(it->second[step]) + " vs "
        + std::to_string(value));
  }
  it->second[step] = value;
}

void addConstantVelocity(const MixedFESpace&            space,
                         const std::set<Index>&         nodes,
                         const BCsParams&               bc,
                         Index                          steps,
                         std::map<Index, Vector<Real>>& values)
{
  const auto                               u_dof = space.field(0);
  const std::array<std::optional<Real>, 3> comps = {bc.ux, bc.uy, bc.uz};
  for (Index node : nodes)
  {
    for (Index comp = 0; comp < u_dof.numComponents(); ++comp)
    {
      if (!comps[static_cast<std::size_t>(comp)])
      {
        continue;
      }
      const Index dof   = u_dof.globalDof(node, comp);
      const Real  value = *comps[static_cast<std::size_t>(comp)];
      for (Index step = 0; step < steps; ++step)
      {
        addBoundaryValue(values, dof, step, steps, value);
      }
    }
  }
}

void addConstantPressure(const MixedFESpace&            space,
                         const std::set<Index>&         nodes,
                         const BCsParams&               bc,
                         Index                          steps,
                         std::map<Index, Vector<Real>>& values)
{
  if (!bc.p)
  {
    return;
  }

  const auto p_dof = space.field(1);
  for (Index node : nodes)
  {
    const Index dof = p_dof.globalDof(node, 0);
    for (Index step = 0; step < steps; ++step)
    {
      addBoundaryValue(values, dof, step, steps, *bc.p);
    }
  }
}

void addProfileVelocity(const MixedFESpace&            space,
                        const BCsParams&               bc,
                        Index                          steps,
                        Real                           dt,
                        std::map<Index, Vector<Real>>& values)
{
  if (!bc.velocity)
  {
    return;
  }

  const DirichletControl boundary =
      makeVelocityControl(space, bcSelector(bc));
  for (Index step = 0; step < steps; ++step)
  {
    for (Index i = 0; i < boundary.numDofs(); ++i)
    {
      addBoundaryValue(values,
                       boundary.stateDof(i),
                       step,
                       steps,
                       trueValue(space, boundary, *bc.velocity, step, i, dt));
    }
  }
}

BoundaryValues makeBoundaryValues(const MixedFESpace&  space,
                                  const ForwardParams& forward)
{
  const Index steps = forward.time.steps;
  const Real  dt    = forward.time.dt;

  std::map<Index, Vector<Real>> values;
  for (const BCsParams& bc : forward.bcs)
  {
    const BoundarySelector selector = bcSelector(bc);
    const std::set<Index>  nodes    = boundaryNodes(space.mesh(), selector);
    addProfileVelocity(space, bc, steps, dt, values);
    addConstantVelocity(space, nodes, bc, steps, values);
    addConstantPressure(space, nodes, bc, steps, values);
  }

  if (values.empty())
  {
    throw std::runtime_error("make-obs found no Dirichlet boundary values");
  }

  BoundaryValues out;
  out.dofs.reserve(static_cast<Index>(values.size()));
  for (const auto& entry : values)
  {
    out.dofs.push_back(entry.first);
  }

  out.prm.resize(steps * out.dofs.size());
  Index i = 0;
  for (const auto& entry : values)
  {
    for (Index step = 0; step < steps; ++step)
    {
      if (std::isnan(entry.second[step]))
      {
        throw std::runtime_error(
            "make-obs boundary value was not assigned for every time step");
      }
      out.prm[step * out.dofs.size() + i] = entry.second[step];
    }
    ++i;
  }
  return out;
}

const TargetParams& reynoldsTarget(const ForwardParams& forward)
{
  for (const auto& bc : forward.bcs)
  {
    if (bc.velocity)
    {
      return *bc.velocity;
    }
  }
  throw std::runtime_error(
      "forward.fluid.reynolds requires at least one velocity boundary");
}

FluidParams makeFluidParams(const ForwardParams& forward)
{
  FluidParams fluid;
  fluid.rho = forward.fluid.rho;
  if (forward.fluid.mu)
  {
    fluid.mu = *forward.fluid.mu;
    return fluid;
  }

  const TargetParams& target = reynoldsTarget(forward);
  fluid.mu                   = forward.fluid.rho * target.bulk_speed * 2.0 * target.radius
             / *forward.fluid.reynolds;
  return fluid;
}

struct NoiseReport
{
  bool  enabled    = false;
  Real  signal_rms = 0.0;
  Real  sigma      = 0.0;
  Index count      = 0;
};

Real snrAmplitudeRatio(const NoiseParams& noise)
{
  if (noise.snr)
  {
    return *noise.snr;
  }
  return std::pow(10.0, *noise.snr_db / 20.0);
}

NoiseReport addGaussianNoise(TimeObservationData& data,
                             const NoiseParams&   noise)
{
  NoiseReport report;
  if (!noise.enabled)
  {
    return report;
  }

  Real  sum_sq = 0.0;
  Index count  = 0;
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    const Vector<Real> values = data[level];
    for (Index i = 0; i < values.size(); ++i)
    {
      sum_sq += values[i] * values[i];
      ++count;
    }
  }
  if (count == 0)
  {
    throw std::runtime_error("Cannot add noise to empty observation data");
  }

  report.enabled    = true;
  report.signal_rms = std::sqrt(sum_sq / static_cast<Real>(count));
  report.count      = count;
  if (report.signal_rms <= 0.0)
  {
    throw std::runtime_error("Cannot set SNR for zero observation signal");
  }
  report.sigma = report.signal_rms / snrAmplitudeRatio(noise);

  std::mt19937_64                rng(kNoiseSeed);
  std::normal_distribution<Real> dist(0.0, report.sigma);
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    Vector<Real> values = data[level];
    for (Index i = 0; i < values.size(); ++i)
    {
      values[i] += dist(rng);
    }
  }
  return report;
}

void ensureParentDir(const std::string& path)
{
  const std::filesystem::path file(path);
  const std::filesystem::path dir = file.parent_path();
  if (!dir.empty())
  {
    std::filesystem::create_directories(dir);
  }
}

bool hasTimeSample(const TimeSampleParams& time)
{
  return time.start_time || time.end_time || time.start_level
         || time.end_level || time.num_points;
}

Real startTime(const TimeSampleParams& time,
               Real                    dt)
{
  if (time.start_time)
  {
    return *time.start_time;
  }
  if (time.start_level)
  {
    return static_cast<Real>(*time.start_level) * dt;
  }
  return 0.0;
}

Real endTime(const TimeSampleParams& time,
             Real                    dt,
             Index                   max_level)
{
  if (time.end_time)
  {
    return *time.end_time;
  }
  if (time.end_level)
  {
    return static_cast<Real>(*time.end_level) * dt;
  }
  return static_cast<Real>(max_level) * dt;
}

void checkTimeRange(Real start,
                    Real end,
                    Real max_time)
{
  const Real tol =
      std::max<Real>(1.0e-10, 1.0e-8 * std::max<Real>(1.0, max_time));
  if (start < -tol || end > max_time + tol)
  {
    throw std::runtime_error(
        "make_obs.time range is outside the forward trajectory");
  }
  if (start > end)
  {
    throw std::runtime_error("make_obs.time start must not exceed end");
  }
}

Vector<Real> selectedTimes(const TimeSampleParams& time,
                           Real                    dt,
                           Index                   max_level)
{
  const Real begin    = startTime(time, dt);
  const Real finish   = endTime(time, dt, max_level);
  const Real max_time = static_cast<Real>(max_level) * dt;
  checkTimeRange(begin, finish, max_time);

  const Index count =
      time.num_points
          ? *time.num_points
          : static_cast<Index>(std::llround((finish - begin) / dt)) + 1;
  Vector<Real> times(count);
  if (count == 1)
  {
    times[0] = begin;
    return times;
  }

  const Index intervals = count - 1;
  for (Index i = 0; i < count; ++i)
  {
    times[i] = begin
               + (finish - begin) * static_cast<Real>(i)
                     / static_cast<Real>(intervals);
  }
  return times;
}

Vector<Real> relativeTimes(const Vector<Real>& absolute_times,
                           Real                start)
{
  Vector<Real> times(absolute_times.size());
  for (Index i = 0; i < absolute_times.size(); ++i)
  {
    times[i] = absolute_times[i] - start;
    if (i == 0 && std::abs(times[i]) < 1.0e-14)
    {
      times[i] = 0.0;
    }
  }
  return times;
}

TimeObservationData interpolateTimeObs(const TimeObservationData& data,
                                       const TimeSampleParams&    time,
                                       Real                       dt)
{
  if (!hasTimeSample(time))
  {
    return data;
  }

  const Real          begin = startTime(time, dt);
  const Vector<Real>  times = selectedTimes(time, dt, data.numLevels() - 1);
  TimeObservationData out(times.size(), data.numObservations());
  out.setLayout(data.sampler(), data.points(), data.components());
  out.setTimeValues(relativeTimes(times, begin));

  for (Index row = 0; row < times.size(); ++row)
  {
    const Real  scaled       = times[row] / dt;
    const Index lower        = static_cast<Index>(std::floor(scaled));
    const Index upper        = std::min<Index>(lower + 1, data.numLevels() - 1);
    const Real  upper_weight = scaled - static_cast<Real>(lower);

    const Vector<Real> lo  = data[lower];
    const Vector<Real> hi  = data[upper];
    Vector<Real>       dst = out[row];
    for (Index i = 0; i < data.numObservations(); ++i)
    {
      dst[i] = (1.0 - upper_weight) * lo[i] + upper_weight * hi[i];
    }
  }
  return out;
}

int run(const make_obs::Params& prm)
{
  const ForwardParams& forward = prm.forward;
  const Index          steps   = forward.time.steps;
  requireReSolve(forward.solver);

  Mesh mesh = GmshReader::read(forward.mesh.file);

  auto         elem  = makeElement(mesh);
  MixedFESpace space = makeSpace(mesh, *elem);
  std::cout << FEMX_MAKE_OBS_APP_NAME << ": ranks = 1"
            << ", dofs = " << space.numDofs()
            << ", cells = " << space.mesh().numElems() << '\n';

  TimeNavierStokesParameters ns_prm;
  ns_prm.steps      = steps;
  ns_prm.dt         = forward.time.dt;
  ns_prm.fluid      = makeFluidParams(forward);
  ns_prm.quad_order = kQuadOrder;

  NavierStokesEquation ns_eq(space, ns_prm);

  const BoundaryValues bc_values = makeBoundaryValues(space, forward);
  DirichletControl     dirichlet(bc_values.dofs);

  DirichletControlEquation eq(ns_eq, dirichlet);

  const CsrPattern pattern = SparsityPatternBuilder::build(space);

  SparseSystemMatrix  next_jac(pattern);
  ReSolveLinearSolver lin_solver(
      workspaceType(forward.solver), makeReSolveOptions());

  TimeMatrixLinearStateSolver state_solver(eq, next_jac, lin_solver);

  Vector<Real> x_init(eq.numStates());
  x_init.setZero();
  state_solver.setInitialState(x_init);

  ProgressPrinter progress;
  state_solver.setStepMonitor(
      [&progress](Index step, Index total)
      {
        progress.timeStep(step, total);
      });

  TimeStateTrajectory tr;
  progress.beginPhase("forward solve");
  state_solver.solve(bc_values.prm, tr);
  progress.finishLine();

  writeForwardViz(mesh,
                  space,
                  tr,
                  forward.time.dt,
                  {forward.output.basename});

  auto obs = makeObs(space,
                     prm.obs,
                     steps,
                     eq.numStates(),
                     eq.numParams());

  progress.beginPhase("sample observations");
  TimeObservationData sampled = sampleTimeObs(*obs, tr, bc_values.prm);
  setObsLayout(sampled, space, prm.obs);
  sampled = interpolateTimeObs(sampled, prm.time, forward.time.dt);
  TimeObservationData noisy = sampled;
  const NoiseReport   noise = addGaussianNoise(noisy, prm.noise);

  ensureParentDir(prm.output.file);
  writeTimeObsData(prm.output.file, noisy);
  const std::string vti_output = writeObservationVtiOutputs(prm, noisy);

  std::cout << "\nFinal summary\n";
  std::cout << "  levels: " << noisy.numLevels()
            << ", observations/level: " << noisy.numObservations() << '\n';
  std::cout << "  output: " << prm.output.file << '\n';
  if (!vti_output.empty())
  {
    std::cout << "  vti output: " << vti_output << '\n';
  }
  if (noise.enabled)
  {
    std::cout << "  noise: gaussian, signal_rms = " << noise.signal_rms
              << ", sigma = " << noise.sigma
              << ", sampled values = " << noise.count
              << ", seed = " << kNoiseSeed << '\n';
  }
  else
  {
    std::cout << "  noise: disabled\n";
  }
  std::cout << "  TIMING forward_assembly_s="
            << state_solver.assemblySeconds()
            << " forward_solve_s=" << state_solver.solveSeconds()
            << " forward_assembly_calls=" << state_solver.assemblyCalls()
            << " forward_solve_calls=" << state_solver.solveCalls()
            << '\n';

  return 0;
}

} // namespace

int main(int argc, char** argv)
{
  int exit_code = 0;
  try
  {
    const AppOptions options = parseAppOptions(argc, argv);
    if (options.help)
    {
      printUsage(std::cout);
      return 0;
    }
    if (options.config_file.empty())
    {
      throw std::runtime_error("--config FILE is required");
    }

    make_obs::Params prm = make_obs::loadConfig(options.config_file);
    applyOverrides(options, prm);
    exit_code = run(prm);
  }
  catch (const std::exception& e)
  {
    std::cerr << FEMX_MAKE_OBS_APP_NAME << " failed: " << e.what()
              << '\n';
    exit_code = 1;
  }
  return exit_code;
}
