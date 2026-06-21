#include "RunSupport.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

#include "BoundaryConditions.hpp"
#include <NavierHelper.hpp>
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/solve/TimeTrajectory.hpp>

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

navier::NavierKernel makeKernel(const MixedFESpace&    space,
                                const GaussQuadrature& quad,
                                const FluidParams&     fluid,
                                Real                   dt)
{
  return navier::makeNavierKernel(space.field(0).space(),
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
    throw std::runtime_error("mesh file is required");
  }
  if (prm.time.steps <= 0 || prm.time.dt <= 0.0)
  {
    throw std::runtime_error("time steps and dt must be positive");
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
  if (bc.dofs().size() != bc.values().size())
  {
    throw std::runtime_error("DirichletCondition has inconsistent data");
  }

  for (Index i = 0; i < bc.dofs().size(); ++i)
  {
    const Index dof = bc.dofs()[i];
    if (dof < 0 || dof >= state.size())
    {
      throw std::runtime_error("Dirichlet dof is out of range");
    }
    state[dof] = bc.values()[i];
  }
}

Vector<Real> makeInitialState(const MixedFESpace&           space,
                              const std::vector<BCsParams>& bcs)
{
  Vector<Real> state(space.numDofs());
  state.setZero();
  assignBoundaryValues(state, makeBoundaryCondition(space, bcs, 0.0));
  return state;
}

FixedBoundaryValues toFixedBoundaryValues(
    const std::map<Index, Vector<Real>>& values,
    Index                                steps)
{
  FixedBoundaryValues out;
  out.dofs.reserve(static_cast<Index>(values.size()));
  for (const auto& entry : values)
  {
    out.dofs.push_back(entry.first);
  }

  out.values.resize(steps * out.dofs.size());
  Index i = 0;
  for (const auto& entry : values)
  {
    for (Index step = 0; step < steps; ++step)
    {
      if (std::isnan(entry.second[step]))
      {
        throw std::runtime_error(
            "fixed boundary value was not assigned for every time step");
      }
      out.values[step * out.dofs.size() + i] = entry.second[step];
    }
    ++i;
  }
  return out;
}

FixedBoundaryValues makeFixedBoundaryValues(
    const MixedFESpace&           space,
    const std::vector<BCsParams>& bcs,
    Index                         steps,
    Real                          dt)
{
  if (steps <= 0)
  {
    throw std::runtime_error("fixed boundary values require positive steps");
  }

  const Real                    unset = std::numeric_limits<Real>::quiet_NaN();
  std::map<Index, Vector<Real>> values;

  for (Index step = 0; step < steps; ++step)
  {
    const Real time = static_cast<Real>(step + 1) * dt;
    const auto bc   = makeBoundaryCondition(space, bcs, time);
    if (bc.dofs().size() != bc.values().size())
    {
      throw std::runtime_error("DirichletCondition has inconsistent data");
    }

    for (Index i = 0; i < bc.dofs().size(); ++i)
    {
      Vector<Real>& series = values[bc.dofs()[i]];
      if (series.empty())
      {
        series.resize(steps);
        for (Index k = 0; k < steps; ++k)
        {
          series[k] = unset;
        }
      }
      series[step] = bc.values()[i];
    }
  }

  return toFixedBoundaryValues(values, steps);
}

} // namespace

double elapsedSeconds(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double>(end - begin).count();
}

ForwardProblem::ForwardProblem(const Params& prm)
  : steps(prm.time.steps),
    dt(prm.time.dt),
    mesh(readProblemMesh(prm)),
    elem(makeElem(mesh, "ns-gls")),
    space(navier::makeSpace(mesh, *elem)),
    quad(makeVelocityQuadrature(space)),
    ns(makeKernel(space, quad, prm.fluid, dt)),
    fem(steps, DofLayout(space), DofLayout(space), DofLayout(space), ns),
    fixed(makeFixedBoundaryValues(space, prm.bcs, steps, dt)),
    eq(fem, DirichletControl{}, fixed.dofs, 0, 0, fixed.values),
    x0(makeInitialState(space, prm.bcs)),
    pattern(assembly::SparsityPatternBuilder::build(space)),
    prm0(0)
{
}

AppOptions parseAppOptions(int   argc,
                           char* argv[],
                           bool  allow_unknown_options)
{
  AppOptions options;

  const auto requireValue = [argc, argv](int& i, const std::string& key)
  {
    if (i + 1 >= argc)
    {
      throw std::runtime_error("Missing value for " + key);
    }
    return std::string(argv[++i]);
  };

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
      options.config_file = requireValue(i, key);
      continue;
    }
    if (key == "--steps")
    {
      options.steps = static_cast<Index>(
          std::stoi(requireValue(i, key)));
      if (*options.steps <= 0)
      {
        throw std::runtime_error("--steps must be positive");
      }
      continue;
    }
    if (key == "--no-output")
    {
      options.no_output = true;
      continue;
    }
    if (!allow_unknown_options)
    {
      throw std::runtime_error("Unknown option: " + key);
    }
  }

  if (options.config_file.empty())
  {
    throw std::runtime_error("Missing required option: --config FILE");
  }

  return options;
}

void printUsage(std::ostream&                   out,
                const std::string&              executable,
                const std::string&              option_suffix,
                const std::vector<std::string>& extra_lines)
{
  out << "Usage: " << executable << " --config FILE" << option_suffix << '\n'
      << "       " << executable
      << " --config FILE --steps N --no-output" << option_suffix << '\n';
  for (const std::string& line : extra_lines)
  {
    out << line << '\n';
  }
}

std::unique_ptr<FiniteElement> makeElem(const Mesh&        mesh,
                                        const std::string& executable)
{
  (void) executable;
  try
  {
    return navier::makeElement(mesh);
  }
  catch (const std::runtime_error& e)
  {
    throw std::runtime_error(std::string(e.what()) + " (" + executable + ")");
  }
}

bool isFinite(const Vector<Real>& x)
{
  for (Index i = 0; i < x.size(); ++i)
  {
    if (!std::isfinite(x[i]))
    {
      return false;
    }
  }
  return true;
}

bool shouldWriteOutput(Index               step,
                       Index               num_steps,
                       const OutputParams& prm)
{
  return step % prm.interval == 0 || step == num_steps;
}

namespace
{

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

} // namespace

Snapshot makeSnapshot(const MixedFESpace& space,
                      const Vector<Real>& x,
                      Real                time)
{
  const Index nodes = space.mesh().numNodes();
  Snapshot    snapshot{
      time, Vector<Real>(nodes), Vector<Real>(nodes), Vector<Real>(nodes), Vector<Real>(nodes)};
  splitFields(x, space, snapshot.ux, snapshot.uy, snapshot.uz, snapshot.p);
  return snapshot;
}

void writeOutput(const Mesh&                  mesh,
                 const OutputParams&          prm,
                 const std::vector<Snapshot>& snapshots)
{
  std::filesystem::create_directories(prm.directory);

  TimeSeriesDataOut vel_out;
  vel_out.attachMesh(mesh);

  TimeSeriesDataOut pre_out;
  pre_out.attachMesh(mesh);

  for (const Snapshot& snap : snapshots)
  {
    vel_out.beginStep(snap.time);
    vel_out.addNodalVectorField("velocity",
                                snap.ux,
                                snap.uy,
                                snap.uz);

    pre_out.beginStep(snap.time);
    pre_out.addNodalScalarField("pressure", snap.p);
  }

  vel_out.write(prm.directory + "/velocity");
  pre_out.write(prm.directory + "/pressure");
}

void writeTrajectoryOutput(const ForwardProblem&        problem,
                           const solve::TimeTrajectory& tr,
                           const OutputParams&          prm)
{
  std::vector<Snapshot> snapshots;
  for (Index step = 1; step <= problem.steps; ++step)
  {
    if (shouldWriteOutput(step, problem.steps, prm))
    {
      snapshots.push_back(makeSnapshot(
          problem.space, tr[step], static_cast<Real>(step) * problem.dt));
    }
  }

  if (!snapshots.empty())
  {
    writeOutput(problem.mesh, prm, snapshots);
  }
}

void writeBuildInfo(const OutputParams& prm, const BuildInfo& info)
{
  std::filesystem::create_directories(prm.directory);

  std::ofstream out(prm.directory + "/build-info.txt");
  if (!out)
  {
    throw std::runtime_error("Failed to open build-info.txt for writing");
  }

  for (const auto& [key, value] : info.entries)
  {
    out << key << ": " << value << '\n';
  }
}

std::ofstream openRunLog(const OutputParams& prm)
{
  std::filesystem::create_directories(prm.directory);

  std::ofstream out(prm.directory + "/run-info.txt");
  if (!out)
  {
    throw std::runtime_error("Failed to open run-info.txt for writing");
  }

  return out;
}

} // namespace femx
