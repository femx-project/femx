#include "PoissonProblem.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include <femx/fem/DirichletBC.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/runtime/Cli.hpp>

using namespace femx;
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

PoissonProblem::PoissonProblem(const Options& opts)
  : opts_(opts),
    mesh_(makePoissonMesh(opts)),
    space_(&mesh_, &fe_)
{
  space_.setup();
  elem_data_ = makeElementQuadData(space_, GaussQuadrature::make(fe_.shape(), 2));
  assm_map_  = assembly::makeAssemblyMap(space_.dofMap());

  DirichletBC boundary;
  boundary.addBoundary(space_, onBoundary, boundaryValue);
  boundary_values_ = boundary.vals();
  boundary_map_    = assembly::makeBoundaryMap(boundary.dofs());
}

const Options& PoissonProblem::options() const noexcept
{
  return opts_;
}

const Mesh& PoissonProblem::mesh() const noexcept
{
  return mesh_;
}

const HostElementQuadData&
PoissonProblem::elementData() const noexcept
{
  return elem_data_;
}

const assembly::HostAssemblyMap&
PoissonProblem::assemblyMap() const noexcept
{
  return assm_map_;
}

const assembly::HostBoundaryMap&
PoissonProblem::boundaryMap() const noexcept
{
  return boundary_map_;
}

const HostVector<Real>& PoissonProblem::boundaryValues() const noexcept
{
  return boundary_values_;
}

Index PoissonProblem::numNodes() const noexcept
{
  return mesh_.numNodes();
}

Index PoissonProblem::numDofs() const noexcept
{
  return space_.numDofs();
}

ErrorReport PoissonProblem::errorReport(const HostVector<Real>& x) const
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
    const Real val = x[space_.globalDof(in, 0)];
    const Real err = val - exactValue(mesh_.node(in));

    report.min_val  = std::min(report.min_val, val);
    report.max_val  = std::max(report.max_val, val);
    report.max_err  = std::max(report.max_err, std::abs(err));
    err2_sum       += err * err;
  }

  report.rms_err = std::sqrt(err2_sum / static_cast<Real>(mesh_.numNodes()));

  return report;
}

void PoissonProblem::writeSolution(const HostVector<Real>& x,
                                   const std::string&      base) const
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

  HostVector<Real> result(mesh_.numNodes());
  HostVector<Real> exact(mesh_.numNodes());
  HostVector<Real> error(mesh_.numNodes());
  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    result[in] = x[space_.globalDof(in, 0)];
    exact[in]  = exactValue(mesh_.node(in));
    error[in]  = result[in] - exact[in];
  }

  VtuWriter out;
  out.writePointData(path.string(),
                     mesh_,
                     {{"solution", 1, &result},
                      {"exact", 1, &exact},
                      {"error", 1, &error}});
}

Real PoissonProblem::exactValue(const Mesh::Node& p)
{
  return sin(constants::PI * p[0]) * sinh(constants::PI * p[1]) / sinh(constants::PI);
}

Real PoissonProblem::boundaryValue(const Mesh::Node& p, Real)
{
  if (std::abs(p[1] - 1.0) < boundary_eps)
  {
    return sin(constants::PI * p[0]);
  }
  return 0.0;
}

bool PoissonProblem::onBoundary(const Mesh::Node& p, Real)
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
      opts.num_x_cells = parsePositiveIndex(
          runtime::requireValue(argc, argv, i, arg), arg);
      continue;
    }
    if (arg == "--ny")
    {
      opts.num_y_cells = parsePositiveIndex(
          runtime::requireValue(argc, argv, i, arg), arg);
      continue;
    }
    if (arg == "--output")
    {
      opts.write_output = parseYesNo(
          runtime::requireValue(argc, argv, i, arg), arg);
      continue;
    }
    if (arg == "--backend" || arg == "-b")
    {
      opts.memspace = parseBackend(
          runtime::requireValue(argc, argv, i, arg));
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
            << " [--nx N] [--ny N] [-b cpu|cuda]"
            << " [--output yes|no]";
  if (petsc_options)
  {
    std::cout << " [PETSc options]";
  }
  std::cout << '\n';
  std::cout
      << "  -b, --backend cpu|cuda selects the execution backend";
  if (backend_note)
  {
    std::cout << " (" << backend_note << ")";
  }
  else if (petsc_options)
  {
    std::cout << " (PETSc supports Host execution only)";
  }
  std::cout << '\n';
  std::cout << "  --output yes writes a VTU file under "
            << outputDir()
            << '\n';
}

void printReport(std::ostream&         out,
                 const std::string&    configuration,
                 const PoissonProblem& problem,
                 const ErrorReport&    err,
                 Real                  rnorm)
{
  const Options& opts = problem.options();
  out << "Poisson forward (" << configuration << ")\n";
  out << "  cells: " << opts.num_x_cells << " x " << opts.num_y_cells
      << '\n';
  out << "  nodes: " << problem.numNodes() << '\n';
  out << "  dofs: " << problem.numDofs() << '\n';
  out << "  solution range: [" << err.min_val << ", "
      << err.max_val << "]\n";
  out << "  residual l2 norm: " << rnorm << '\n';
  out << "  rms nodal error: " << err.rms_err << '\n';
  out << "  max nodal error: " << err.max_err << '\n';
}

} // namespace femx::examples::poisson
