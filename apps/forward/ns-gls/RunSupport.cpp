#include "RunSupport.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "BCs.hpp"
#include <NavierHelper.hpp>
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/linalg/BlockVectorView.hpp>
#include <femx/state/TimeLinearStateSolver.hpp>
#include <femx/state/TimeTrajectory.hpp>

using namespace std;
using namespace femx::state;
using namespace femx::assembly;
using namespace femx::navier;

namespace femx
{
namespace
{

constexpr Index kQuadOrder = 2;

GaussQuadrature makeVelocityQuadrature(const MixedFESpace& space)
{
  return GaussQuadrature::make(
      space.field(0).space().finiteElement().referenceElement(), kQuadOrder);
}

NavierKernel makeKernel(const MixedFESpace&    space,
                        const GaussQuadrature& quad,
                        const FluidParams&     fluid,
                        Real                   dt)
{
  return makeNavierKernel(space.field(0).space(),
                          quad,
                          space.numDofsPerElem(),
                          fluid.rho,
                          fluid.mu,
                          dt);
}

void requireForwardParams(const Params& prm)
{
  if (prm.mesh_file.empty())
  {
    throw runtime_error("mesh file is required");
  }
  if (prm.time.steps <= 0 || prm.time.dt <= 0.0)
  {
    throw runtime_error("time steps and dt must be positive");
  }
}

Mesh readProblemMesh(const Params& prm)
{
  requireForwardParams(prm);
  return GmshReader::read(prm.mesh_file);
}

void assignBoundaryValues(Vector<Real>&             state,
                          const DirichletCondition& bc)
{
  if (bc.dofs().size() != bc.vals().size())
  {
    throw runtime_error("DirichletCondition has inconsistent data");
  }

  for (Index i = 0; i < bc.dofs().size(); ++i)
  {
    const Index id = bc.dofs()[i];
    if (id < 0 || id >= state.size())
    {
      throw runtime_error("Dirichlet id is out of range");
    }
    state[id] = bc.vals()[i];
  }
}

Vector<Real> makeInitialState(const MixedFESpace&      space,
                              const Vector<BCsParams>& bcs)
{
  Vector<Real> state(space.numDofs());
  state.setZero();
  assignBoundaryValues(state, makeBoundaryCondition(space, bcs, 0.0));
  return state;
}

FixedBoundaryValues toFixedBoundaryValues(
    const map<Index, Vector<Real>>& vals,
    Index                           steps)
{
  FixedBoundaryValues out;
  for (const auto& entry : vals)
  {
    out.dofs.push_back(entry.first);
  }

  out.vals.resize(steps * out.dofs.size());
  BlockVectorView<Real> values(out.vals.data(), steps, out.dofs.size());
  Index                 i = 0;
  for (const auto& entry : vals)
  {
    for (Index step = 0; step < steps; ++step)
    {
      if (isnan(entry.second[step]))
      {
        throw runtime_error(
            "fixed boundary value was not assigned for every time step");
      }
      values(step, i) = entry.second[step];
    }
    ++i;
  }
  return out;
}

FixedBoundaryValues makeFixedBoundaryValues(
    const MixedFESpace&      space,
    const Vector<BCsParams>& bcs,
    Index                    steps,
    Real                     dt)
{
  if (steps <= 0)
  {
    throw runtime_error("fixed boundary values require positive steps");
  }

  const Real               unset = numeric_limits<Real>::quiet_NaN();
  map<Index, Vector<Real>> vals;

  for (Index step = 0; step < steps; ++step)
  {
    const Real time = static_cast<Real>(step + 1) * dt;
    const auto bc   = makeBoundaryCondition(space, bcs, time);
    if (bc.dofs().size() != bc.vals().size())
    {
      throw runtime_error("DirichletCondition has inconsistent data");
    }

    for (Index i = 0; i < bc.dofs().size(); ++i)
    {
      Vector<Real>& series = vals[bc.dofs()[i]];
      if (series.empty())
      {
        series.resize(steps);
        for (Index k = 0; k < steps; ++k)
        {
          series[k] = unset;
        }
      }
      series[step] = bc.vals()[i];
    }
  }

  return toFixedBoundaryValues(vals, steps);
}

} // namespace

double elapsedSeconds(Clock::time_point begin, Clock::time_point end)
{
  return chrono::duration<double>(end - begin).count();
}

ForwardProblem::ForwardProblem(const Params& prm)
  : steps(prm.time.steps),
    dt(prm.time.dt),
    mesh(readProblemMesh(prm)),
    elem(makeElem(mesh, "ns-gls")),
    space(makeSpace(mesh, *elem)),
    quad(makeVelocityQuadrature(space)),
    ns(makeKernel(space, quad, prm.fluid, dt)),
    fem(steps,
        DofLayout(space),
        Vector<DofLayout>{DofLayout(space), DofLayout(space)},
        DofLayout(space),
        ns),
    fixed(makeFixedBoundaryValues(space, prm.bcs, steps, dt)),
    problem(fem, DirichletControl{}, fixed.dofs, 0, 0, fixed.vals),
    x0(makeInitialState(space, prm.bcs)),
    pettern(SparsityPatternBuilder::build(space)),
    prm0(0)
{
}

AppOptions parseAppOptions(int   argc,
                           char* argv[],
                           bool  allow_unknown_options)
{
  AppOptions opts;

  const auto requireValue = [argc, argv](int& i, const string& key)
  {
    if (i + 1 >= argc)
    {
      throw runtime_error("Missing value for " + key);
    }
    return string(argv[++i]);
  };

  for (int i = 1; i < argc; ++i)
  {
    const string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      opts.help = true;
      return opts;
    }
    if (key == "--config" || key == "-config")
    {
      opts.config_file = requireValue(i, key);
      continue;
    }
    if (key == "--steps")
    {
      opts.steps = static_cast<Index>(
          stoi(requireValue(i, key)));
      if (*opts.steps <= 0)
      {
        throw runtime_error("--steps must be positive");
      }
      continue;
    }
    if (key == "--no-output")
    {
      opts.no_output = true;
      continue;
    }
    if (!allow_unknown_options)
    {
      throw runtime_error("Unknown option: " + key);
    }
  }

  if (opts.config_file.empty())
  {
    throw runtime_error("Missing required option: --config FILE");
  }

  return opts;
}

void printUsage(ostream&              out,
                const string&         executable,
                const string&         option_suffix,
                const Vector<string>& extra_lines)
{
  out << "Usage: " << executable << " --config FILE" << option_suffix << '\n'
      << "       " << executable
      << " --config FILE --steps N --no-output" << option_suffix << '\n';
  for (const string& line : extra_lines)
  {
    out << line << '\n';
  }
}

unique_ptr<FiniteElement> makeElem(const Mesh&   mesh,
                                   const string& executable)
{
  (void) executable;
  try
  {
    return makeElement(mesh);
  }
  catch (const runtime_error& e)
  {
    throw runtime_error(string(e.what()) + " (" + executable + ")");
  }
}

bool isFinite(const Vector<Real>& x)
{
  for (Index i = 0; i < x.size(); ++i)
  {
    if (!isfinite(x[i]))
    {
      return false;
    }
  }
  return true;
}

bool shouldWriteOutput(Index               step,
                       Index               nt,
                       const OutputParams& prm)
{
  return step % prm.interval == 0 || step == nt;
}

namespace
{

Real velocityRelativeChange(const MixedFESpace& space,
                            const Vector<Real>& previous,
                            const Vector<Real>& current)
{
  if (previous.size() != current.size()
      || previous.size() != space.numDofs())
  {
    throw runtime_error("velocity convergence received incompatible states");
  }

  const auto  velocity = space.field(0);
  const Index nodes    = velocity.space().mesh().numNodes();
  const Index comps    = velocity.numComponents();

  Real diff2 = 0.0;
  Real ref2  = 0.0;
  for (Index in = 0; in < nodes; ++in)
  {
    for (Index d = 0; d < comps; ++d)
    {
      const Index id    = velocity.globalDof(in, d);
      const Real  diff  = current[id] - previous[id];
      diff2            += diff * diff;
      ref2             += previous[id] * previous[id];
    }
  }

  if (diff2 <= 0.0)
  {
    return 0.0;
  }
  if (ref2 <= 0.0)
  {
    return numeric_limits<Real>::infinity();
  }
  return sqrt(diff2 / ref2);
}

Real elemMinEdge(const Element& elem)
{
  Real h = numeric_limits<Real>::infinity();
  for (Index i = 0; i < elem.numNodes(); ++i)
  {
    for (Index j = i + 1; j < elem.numNodes(); ++j)
    {
      h = min(h, distance(elem.node(i), elem.node(j)));
    }
  }
  return isfinite(h) ? h : 0.0;
}

Real maxVelocityCfl(const MixedFESpace& space,
                    const Vector<Real>& state,
                    Real                dt)
{
  if (state.size() != space.numDofs())
  {
    throw runtime_error("CFL calculation received incompatible state size");
  }

  const auto  velocity = space.field(0);
  const Index comps    = velocity.numComponents();
  Real        max_cfl  = 0.0;

  for (Index ie = 0; ie < space.mesh().numElems(); ++ie)
  {
    const Element& elem = space.mesh().elem(ie);
    const Real     h    = elemMinEdge(elem);
    if (h <= 0.0)
    {
      continue;
    }

    for (Index in = 0; in < elem.numNodes(); ++in)
    {
      const Index id   = elem.nodeIds()[in];
      Real        vel2 = 0.0;
      for (Index d = 0; d < comps; ++d)
      {
        const Real value  = state[velocity.globalDof(id, d)];
        vel2             += value * value;
      }
      max_cfl = max(max_cfl, sqrt(vel2) * dt / h);
    }
  }

  return max_cfl;
}

struct StepLog
{
  Clock::time_point begin;
  Index             level{0};
  Real              time{0.0};
  Real              max_cfl{0.0};
  Real              vel_change{0.0};
  Real              assm_seconds{0.0};
  Real              solve_seconds{0.0};
  bool              ready{false};
};

struct StopCondition
{
  TimeLinearStateSolver& state_solver;
  const ForwardProblem&  problem;
  const TimeParams&      time;
  bool                   log_steps{false};
  ForwardSolveResult&    result;
  StepLog&               log;

  bool operator()(Index               level,
                  const Vector<Real>& current,
                  const Vector<Real>& previous) const
  {
    if (time.convergence.enabled)
    {
      result.vel_change = velocityRelativeChange(problem.space, previous, current);

      const bool cond_steps = level >= time.convergence.min_steps;
      const bool cond_vel   = result.vel_change < time.convergence.velocity_relative_tolerance;

      result.converged = cond_steps && cond_vel;
    }
    else
    {
      result.converged = false;
    }

    if (log_steps)
    {
      log.level      = level;
      log.time       = static_cast<Real>(level) * problem.dt;
      log.max_cfl    = maxVelocityCfl(problem.space, previous, problem.dt);
      log.vel_change = result.vel_change;
      if (!isfinite(log.max_cfl))
      {
        throw runtime_error("Stopping as CFL became invalid");
      }
      log.assm_seconds  = state_solver.lastAssemblySeconds();
      log.solve_seconds = state_solver.lastSolveSeconds();
      log.ready         = true;
    }
    return result.converged;
  }
};

string stepLogLine(Index step,
                   Real  time,
                   Real  max_cfl,
                   bool  show_velocity_change,
                   Real  vel_change,
                   Real  assembly_seconds,
                   Real  solve_seconds,
                   Real  total_seconds)
{
  ostringstream line;
  line << "step " << setw(7) << step << ", t = " << setw(11) << time
       << ", max CFL = " << setw(11) << max_cfl;
  if (show_velocity_change)
  {
    line << ", rel du = " << setw(11) << vel_change;
  }
  line << ", assembly = " << setw(11) << assembly_seconds << " s"
       << ", solve = " << setw(11) << solve_seconds << " s"
       << ", total = " << setw(11) << total_seconds << " s";
  return line.str();
}

void writeStepLog(const string& line,
                  ostream*      terminal,
                  ostream*      log_out)
{
  if (terminal)
  {
    *terminal << line << '\n';
  }
  if (log_out)
  {
    *log_out << line << '\n';
    log_out->flush();
  }
}

void splitFields(const Vector<Real>& x,
                 const MixedFESpace& space,
                 Vector<Real>&       ux,
                 Vector<Real>&       uy,
                 Vector<Real>&       uz,
                 Vector<Real>&       p)
{
  const Mesh& mesh  = space.mesh();
  const auto  u_dof = space.field(0);
  const auto  p_dof = space.field(1);
  const Index nc    = u_dof.numComponents();

  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    ux[in] = x[u_dof.globalDof(in, 0)];
    uy[in] = 0.0;
    uz[in] = 0.0;
    if (nc > 1)
    {
      uy[in] = x[u_dof.globalDof(in, 1)];
    }
    if (nc > 2)
    {
      uz[in] = x[u_dof.globalDof(in, 2)];
    }
    p[in] = x[p_dof.globalDof(in)];
  }
}

struct OutputSeries
{
  const OutputParams& prm;
  TimeSeriesDataOut   vel_out;
  TimeSeriesDataOut   pre_out;
  Vector<Real>        ux;
  Vector<Real>        uy;
  Vector<Real>        uz;
  Vector<Real>        p;

  OutputSeries(const Mesh& mesh, const OutputParams& prm_in)
    : prm(prm_in),
      ux(mesh.numNodes()),
      uy(mesh.numNodes()),
      uz(mesh.numNodes()),
      p(mesh.numNodes())
  {
    filesystem::create_directories(prm.directory);
    vel_out.attachMesh(mesh);
    pre_out.attachMesh(mesh);
  }

  void add(const MixedFESpace& space,
           const Vector<Real>& state,
           Real                time)
  {
    splitFields(state, space, ux, uy, uz, p);

    vel_out.beginStep(time);
    vel_out.addNodalVectorField("velocity", ux, uy, uz);

    pre_out.beginStep(time);
    pre_out.addNodalScalarField("pressure", p);
  }

  void write() const
  {
    vel_out.write(prm.directory + "/velocity");
    pre_out.write(prm.directory + "/pressure");
  }
};

} // namespace

void writeTrajectoryOutput(const ForwardProblem& problem,
                           const TimeTrajectory& tr,
                           const OutputParams&   prm)
{
  OutputSeries output(problem.mesh, prm);
  for (Index step = 1; step <= problem.steps; ++step)
  {
    if (shouldWriteOutput(step, problem.steps, prm))
    {
      output.add(problem.space,
                 tr[step],
                 static_cast<Real>(step) * problem.dt);
      output.write();
    }
  }
}

namespace
{

struct StateObserver
{
  const ForwardProblem& problem;
  const TimeParams&     time;
  const OutputParams&   prm;
  OutputSeries*         output{nullptr};
  ostream*              terminal{nullptr};
  ostream*              log_out{nullptr};
  ForwardSolveResult&   result;
  StepLog&              log;

  void operator()(Index level, const Vector<Real>& state) const
  {
    if (level > 0)
    {
      result.final_step  = level;
      result.final_time  = static_cast<Real>(level) * problem.dt;
      result.final_state = state;
    }
    if (output && level > 0
        && shouldWriteOutput(level, problem.steps, prm))
    {
      output->add(problem.space, state, result.final_time);
      output->write();
    }
    if (log.ready && log.level == level)
    {
      const Real total_sec = elapsedSeconds(log.begin, Clock::now());
      writeStepLog(stepLogLine(log.level,
                               log.time,
                               log.max_cfl,
                               time.convergence.enabled,
                               log.vel_change,
                               log.assm_seconds,
                               log.solve_seconds,
                               total_sec),
                   terminal,
                   log_out);
      log.begin = Clock::now();
      log.ready = false;
    }
  }
};

} // namespace

ForwardSolveResult solve(TimeLinearStateSolver& state_solver,
                         const ForwardProblem&  problem,
                         const TimeParams&      time,
                         const OutputParams&    prm,
                         bool                   collect_output,
                         ostream*               terminal,
                         ostream*               log_out)
{
  ForwardSolveResult result;
  result.vel_change = numeric_limits<Real>::quiet_NaN();

  optional<OutputSeries> output;
  if (collect_output)
  {
    output.emplace(problem.mesh, prm);
  }

  StepLog log;
  log.begin            = Clock::now();
  const bool log_steps = terminal || log_out;
  if (time.convergence.enabled || log_steps)
  {
    const auto cond = StopCondition{state_solver, problem, time, log_steps, result, log};
    state_solver.setStopCondition(cond);
  }
  else
  {
    state_solver.clearStopCondition();
  }

  const auto observer = StateObserver{problem,
                                      time,
                                      prm,
                                      output ? &(*output) : nullptr,
                                      terminal,
                                      log_out,
                                      result,
                                      log};
  state_solver.solve(problem.prm0, observer);
  state_solver.clearStopCondition();

  if (output && result.final_step > 0
      && !shouldWriteOutput(result.final_step, problem.steps, prm))
  {
    output->add(problem.space, result.final_state, result.final_time);
    output->write();
  }
  return result;
}

void writeBuildInfo(const OutputParams& prm, const BuildInfo& info)
{
  filesystem::create_directories(prm.directory);

  ofstream out(prm.directory + "/build-info.txt");
  if (!out)
  {
    throw runtime_error("Failed to open build-info.txt for writing");
  }

  for (const auto& [key, value] : info.entries)
  {
    out << key << ": " << value << '\n';
  }
}

ofstream openRunLog(const OutputParams& prm)
{
  filesystem::create_directories(prm.directory);

  ofstream out(prm.directory + "/run-info.txt");
  if (!out)
  {
    throw runtime_error("Failed to open run-info.txt for writing");
  }

  return out;
}

} // namespace femx
