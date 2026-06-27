#include "RunSupport.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

#include "BCs.hpp"
#include <ForwardSolveMonitor.hpp>
#include <NavierHelper.hpp>
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/BlockVectorView.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>

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

ForwardSolveResult solve(TimeLinearIntegrator& integrator,
                         const ForwardProblem& problem,
                         const TimeParams&     time,
                         const OutputParams&   prm,
                         bool                  collect_output,
                         ostream*              terminal,
                         ostream*              log_out)
{
  navier::ForwardSolveMonitor monitor(problem.space,
                                      problem.dt,
                                      problem.steps);
  if (collect_output)
  {
    monitor.setFieldOutput(prm.directory, prm.interval);
  }
  if (terminal != nullptr || log_out != nullptr)
  {
    monitor.setDetailedLog(terminal,
                           log_out,
                           time.convergence.enabled);
  }
  monitor.setConvergence(
      {time.convergence.enabled,
       time.convergence.vel_rel_tol,
       time.convergence.min_steps});

  integrator.setMonitor(&monitor);
  integrator.solve(problem.prm0);

  integrator.clearMonitor();
  return monitor.result();
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
