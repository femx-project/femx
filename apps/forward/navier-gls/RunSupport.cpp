#include "RunSupport.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#include <femx/fem/FESpace.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/fem/elements/LagrangeTriangleP1.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/mesh/Mesh.hpp>

namespace femx
{

double elapsedSeconds(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double>(end - begin).count();
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
  if (mesh.numElems() == 0)
  {
    throw std::runtime_error("Mesh has no cells");
  }

  const Cell::Shape shape = mesh.cells().front().shape();
  if (shape == Cell::Shape::Triangle)
  {
    return std::make_unique<LagrangeTriangleP1>();
  }
  if (shape == Cell::Shape::Quadrilateral)
  {
    return std::make_unique<LagrangeQuadQ1>();
  }
  if (shape == Cell::Shape::Tetrahedron)
  {
    return std::make_unique<LagrangeTetrahedronP1>();
  }
  throw std::runtime_error("Unsupported mesh cell type for " + executable);
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
                       const OutputParams& params)
{
  return step % params.interval == 0 || step == num_steps;
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
                 const OutputParams&          params,
                 const std::vector<Snapshot>& snapshots)
{
  std::filesystem::create_directories(params.directory);

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

  vel_out.write(params.directory + "/velocity");
  pre_out.write(params.directory + "/pressure");
}

void writeBuildInfo(const OutputParams& params, const BuildInfo& info)
{
  std::filesystem::create_directories(params.directory);

  std::ofstream out(params.directory + "/build-info.txt");
  if (!out)
  {
    throw std::runtime_error("Failed to open build-info.txt for writing");
  }

  for (const auto& [key, value] : info.entries)
  {
    out << key << ": " << value << '\n';
  }
}

std::ofstream openRunLog(const OutputParams& params)
{
  std::filesystem::create_directories(params.directory);

  std::ofstream out(params.directory + "/run-info.txt");
  if (!out)
  {
    throw std::runtime_error("Failed to open run-info.txt for writing");
  }

  return out;
}

} // namespace femx
