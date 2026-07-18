#include "PoissonForward.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "PoissonOperator.hpp"
#include <femx/assembly/Assembly.hpp>
#include <femx/fem/DirichletBC.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/io/VtuWriter.hpp>

using namespace femx;
using namespace femx::assembly;
using namespace femx::fem;
using namespace femx::io;

#ifndef FEMX_POISSON_DEFAULT_OUTPUT_DIR
#define FEMX_POISSON_DEFAULT_OUTPUT_DIR "output"
#endif

namespace femx::examples::poisson
{
namespace
{

constexpr Real boundary_eps = 1.0e-12;

Index readIndexOption(int& i, int argc, char** argv, const std::string& name)
{
  if (i + 1 >= argc)
  {
    throw std::runtime_error(name + " requires a value");
  }

  const long value = std::stol(argv[++i]);
  if (value <= 0 || value > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(name + " must be a positive integer");
  }
  return static_cast<Index>(value);
}

bool readIndexAssignment(const std::string& arg,
                         const std::string& name,
                         Index&             out)
{
  const std::string prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0)
  {
    return false;
  }

  const long value = std::stol(arg.substr(prefix.size()));
  if (value <= 0 || value > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(name + " must be a positive integer");
  }
  out = static_cast<Index>(value);
  return true;
}

std::string readStringOption(int& i, int argc, char** argv, const std::string& name)
{
  if (i + 1 >= argc)
  {
    throw std::runtime_error(name + " requires a value");
  }
  return argv[++i];
}

bool readStringAssignment(const std::string& arg,
                          const std::string& name,
                          std::string&       out)
{
  const std::string prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0)
  {
    return false;
  }
  out = arg.substr(prefix.size());
  if (out.empty())
  {
    throw std::runtime_error(name + " must not be empty");
  }
  return true;
}

std::string lowerAscii(std::string value)
{
  transform(value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char ch)
            { return static_cast<char>(tolower(ch)); });
  return value;
}

MemorySpace parseMemorySpace(const std::string& value)
{
  const std::string backend = lowerAscii(value);
  if (backend == "cpu")
  {
    return MemorySpace::Host;
  }
  if (backend == "cuda" || backend == "gpu")
  {
    return MemorySpace::Device;
  }
  throw std::runtime_error("Backend must be 'cpu' or 'cuda'");
}

bool parseOutputValue(const std::string& value)
{
  const std::string output = lowerAscii(value);
  if (output == "yes" || output == "true" || output == "on" || output == "1")
  {
    return true;
  }
  if (output == "no" || output == "false" || output == "off" || output == "0")
  {
    return false;
  }
  throw std::runtime_error("--output expects 'yes' or 'no'");
}

MemorySpace readBackendOption(int&               i,
                              int                argc,
                              char**             argv,
                              const std::string& name)
{
  return parseMemorySpace(readStringOption(i, argc, argv, name));
}

bool readBackendAssignment(const std::string& arg,
                           const std::string& name,
                           MemorySpace&       out)
{
  std::string value;
  if (!readStringAssignment(arg, name, value))
  {
    return false;
  }
  out = parseMemorySpace(value);
  return true;
}

Mesh makePoissonMesh(const Options& opts)
{
  if (opts.num_x_cells <= 0 || opts.num_y_cells <= 0)
  {
    throw std::runtime_error("Poisson mesh dimensions must be positive");
  }
  return Mesh::makeStructuredQuad(opts.num_x_cells, opts.num_y_cells);
}

std::filesystem::path vtuPathFromBase(const std::string& base)
{
  std::filesystem::path path(base);
  if (path.extension() == ".vtu")
  {
    return path;
  }
  path += ".vtu";
  return path;
}

} // namespace

PoissonForwardProblem::PoissonForwardProblem(const Options& opts)
  : opts_(opts),
    mesh_(makePoissonMesh(opts)),
    space_(&mesh_, &fe_)
{
  space_.setup();
  geom_ = makeGeometry(mesh_);
  map_  = assembly::makeAssemblyMap(DofLayout(space_));

  DirichletBC boundary;
  boundary.addBoundary(space_, onBoundary, boundaryValue);
  bc_vals_ = boundary.vals();
  bc_plan_ = assembly::makeBoundaryPlan(boundary.dofs(),
                                        map_.graph());
}

const Options& PoissonForwardProblem::options() const noexcept
{
  return opts_;
}

const HostGeometry& PoissonForwardProblem::geom() const noexcept
{
  return geom_;
}

const assembly::HostAssemblyMap&
PoissonForwardProblem::map() const noexcept
{
  return map_;
}

const assembly::HostBoundaryPlan&
PoissonForwardProblem::bcPlan() const noexcept
{
  return bc_plan_;
}

const HostVector& PoissonForwardProblem::bcVals() const noexcept
{
  return bc_vals_;
}

Index PoissonForwardProblem::numNodes() const noexcept
{
  return mesh_.numNodes();
}

Index PoissonForwardProblem::numDofs() const noexcept
{
  return space_.numDofs();
}

void PoissonForwardProblem::assemble(HostCsrMatrix& mat,
                                     HostVector&    rhs) const
{
  HostVector zero_state(numDofs(), 0.0);
  HostVector res;
  CpuContext ctx;
  assembly::assemble(PoissonQuadQ1Operator{},
                     geom_,
                     map_,
                     zero_state,
                     res,
                     mat,
                     ctx);

  rhs.resize(res.size());
  for (Index row = 0; row < res.size(); ++row)
  {
    rhs[row] = -res[row];
  }
  assembly::prepareForwardSolve(bc_plan_, mat, rhs, bc_vals_);
}

ErrorReport PoissonForwardProblem::errorReport(const HostVector& x) const
{
  if (x.size() != space_.numDofs())
  {
    throw std::runtime_error("Poisson solution vector has incompatible size");
  }

  ErrorReport report;
  report.min_val = std::numeric_limits<Real>::infinity();
  report.max_val = -std::numeric_limits<Real>::infinity();
  report.max_err = 0.0;

  Real err2_sum = 0.0;
  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    const Real value = x[space_.globalDof(in, 0)];
    const Real err   = value - exactValue(mesh_.node(in));

    report.min_val  = std::min(report.min_val, value);
    report.max_val  = std::max(report.max_val, value);
    report.max_err  = std::max(report.max_err, std::abs(err));
    err2_sum       += err * err;
  }

  report.rms_err = std::sqrt(err2_sum / static_cast<Real>(mesh_.numNodes()));

  return report;
}

void PoissonForwardProblem::writeSolution(const HostVector&  x,
                                          const std::string& base) const
{
  if (base.empty())
  {
    return;
  }
  if (x.size() != space_.numDofs())
  {
    throw std::runtime_error("Poisson solution vector has incompatible size");
  }

  const std::filesystem::path path = vtuPathFromBase(base);
  if (path.has_parent_path())
  {
    std::filesystem::create_directories(path.parent_path());
  }

  HostVector solution(mesh_.numNodes());
  HostVector exact(mesh_.numNodes());
  HostVector error(mesh_.numNodes());
  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    solution[in] = x[space_.globalDof(in, 0)];
    exact[in]    = exactValue(mesh_.node(in));
    error[in]    = solution[in] - exact[in];
  }

  VtuWriter out;
  out.writePointData(path.string(),
                     mesh_,
                     {{"solution", 1, &solution},
                      {"exact", 1, &exact},
                      {"error", 1, &error}});
}

Real PoissonForwardProblem::exactValue(const Mesh::Node& p)
{
  return sin(constants::PI * p[0]) * sinh(constants::PI * p[1]) / sinh(constants::PI);
}

Real PoissonForwardProblem::boundaryValue(const Mesh::Node& p, Real)
{
  if (std::abs(p[1] - 1.0) < boundary_eps)
  {
    return sin(constants::PI * p[0]);
  }
  return 0.0;
}

bool PoissonForwardProblem::onBoundary(const Mesh::Node& p, Real)
{
  return std::abs(p[0]) < boundary_eps || std::abs(p[0] - 1.0) < boundary_eps
         || std::abs(p[1]) < boundary_eps
         || std::abs(p[1] - 1.0) < boundary_eps;
}

Options parseOptions(int argc, char** argv, bool ignore_unknown)
{
  Options opts;

  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
    {
      continue;
    }
    if (arg == "--nx")
    {
      opts.num_x_cells = readIndexOption(i, argc, argv, arg);
      continue;
    }
    if (arg == "--ny")
    {
      opts.num_y_cells = readIndexOption(i, argc, argv, arg);
      continue;
    }
    if (arg == "--output")
    {
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("-", 0) != 0)
      {
        opts.write_output = parseOutputValue(argv[++i]);
      }
      else
      {
        opts.write_output = true;
      }
      continue;
    }
    if (arg == "--backend" || arg == "-b")
    {
      opts.backend = readBackendOption(i, argc, argv, arg);
      continue;
    }
    if (readIndexAssignment(arg, "--nx", opts.num_x_cells)
        || readIndexAssignment(arg, "--ny", opts.num_y_cells))
    {
      continue;
    }
    if (arg.rfind("--output=", 0) == 0)
    {
      opts.write_output = parseOutputValue(arg.substr(std::string("--output=").size()));
      continue;
    }
    if (arg == "-o" || arg.rfind("-o=", 0) == 0)
    {
      throw std::runtime_error("Use --output yes or --output no");
    }
    if (readBackendAssignment(arg, "--backend", opts.backend)
        || readBackendAssignment(arg, "-b", opts.backend))
    {
      continue;
    }
    if (!ignore_unknown)
    {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  return opts;
}

const char* outputDir()
{
  return FEMX_POISSON_DEFAULT_OUTPUT_DIR;
}

std::string outputStem(const Options& opts)
{
  return std::string("poisson-nx")
         + std::to_string(opts.num_x_cells)
         + "-ny"
         + std::to_string(opts.num_y_cells);
}

void printUsage(const char* app_name,
                bool        petsc_options,
                const char* backend_note)
{
  std::cout << "Usage: " << app_name
            << " [--nx N] [--ny N] [-b cpu|cuda] [--output yes|no]";
  if (petsc_options)
  {
    std::cout << " [PETSc options]";
  }
  std::cout << '\n';
  std::cout << "  -b, --backend cpu|cuda selects the device backend";
  if (backend_note)
  {
    std::cout << " (" << backend_note << ")";
  }
  else if (petsc_options)
  {
    std::cout << " (PETSc supports cpu only)";
  }
  std::cout << '\n';
  std::cout << "  --output yes writes a VTU file under "
            << outputDir()
            << '\n';
}

void printReport(std::ostream&                out,
                 const std::string&           backend,
                 const PoissonForwardProblem& problem,
                 const ErrorReport&           error,
                 Real                         res_norm)
{
  const Options& opts = problem.options();
  out << "Poisson forward (" << backend << ")\n";
  out << "  cells: " << opts.num_x_cells << " x " << opts.num_y_cells
      << '\n';
  out << "  nodes: " << problem.numNodes() << '\n';
  out << "  dofs: " << problem.numDofs() << '\n';
  out << "  solution range: [" << error.min_val << ", "
      << error.max_val << "]\n";
  out << "  residual l2 norm: " << res_norm << '\n';
  out << "  rms nodal error: " << error.rms_err << '\n';
  out << "  max nodal error: " << error.max_err << '\n';
}

} // namespace femx::examples::poisson
