#include "PoissonForward.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <femx/assembly/Assembler.hpp>
#include <femx/fem/BoundaryCondition.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/linalg/AssemblyMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/native/CsrAssemblyMatrix.hpp>
#include <femx/linalg/native/DenseAssemblyMatrix.hpp>

using namespace femx;
using namespace femx::assembly;
using namespace femx::fem;
using namespace femx::io;
using namespace femx::linalg;

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

WorkspaceType parseWorkspaceType(const std::string& value)
{
  const std::string backend = lowerAscii(value);
  if (backend == "cpu")
  {
    return WorkspaceType::Cpu;
  }
  if (backend == "cuda" || backend == "gpu")
  {
    return WorkspaceType::Cuda;
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

WorkspaceType readBackendOption(int&               i,
                                int                argc,
                                char**             argv,
                                const std::string& name)
{
  return parseWorkspaceType(readStringOption(i, argc, argv, name));
}

bool readBackendAssignment(const std::string& arg,
                           const std::string& name,
                           WorkspaceType&     out)
{
  std::string value;
  if (!readStringAssignment(arg, name, value))
  {
    return false;
  }
  out = parseWorkspaceType(value);
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

std::filesystem::path vtuPathFromBase(const std::string& output_base)
{
  std::filesystem::path path(output_base);
  if (path.extension() == ".vtu")
  {
    return path;
  }
  path += ".vtu";
  return path;
}

void applyDenseBoundary(const BoundaryCondition& bc,
                        DenseAssemblyMatrix&     A,
                        Vector<Real>&            rhs)
{
  DenseMatrix& mat = A.mat();
  if (mat.rows() != mat.cols() || rhs.size() != mat.rows())
  {
    throw std::runtime_error(
        "Poisson dense boundary condition received inconsistent dimensions");
  }
  if (bc.dofs().size() != bc.vals().size())
  {
    throw std::runtime_error("DirichletCondition has inconsistent data");
  }

  const Index  size = mat.rows();
  Vector<char> is_dirichlet(size, 0);
  Vector<Real> dirichlet_values(size);

  for (Index c = 0; c < bc.dofs().size(); ++c)
  {
    const Index id = bc.dofs()[c];
    if (id < 0 || id >= size)
    {
      throw std::runtime_error("Dirichlet id is out of range");
    }

    is_dirichlet[id]     = 1;
    dirichlet_values[id] = bc.vals()[c];
  }

  for (Index row = 0; row < size; ++row)
  {
    const bool row_is_dirichlet = is_dirichlet[row] != 0;

    for (Index col = 0; col < size; ++col)
    {
      if (row_is_dirichlet)
      {
        mat(row, col) = row == col ? 1.0 : 0.0;
      }
      else if (is_dirichlet[col] != 0)
      {
        rhs[row]      -= mat(row, col) * dirichlet_values[col];
        mat(row, col)  = 0.0;
      }
    }

    if (row_is_dirichlet)
    {
      rhs[row] = dirichlet_values[row];
    }
  }
}

void applyBoundary(const BoundaryCondition& bc,
                   AssemblyMatrix&          A,
                   Vector<Real>&            rhs)
{
  if (auto* csr = dynamic_cast<CsrAssemblyMatrix*>(&A))
  {
    bc.apply(csr->mat(), rhs);
    return;
  }

  if (auto* dense = dynamic_cast<DenseAssemblyMatrix*>(&A))
  {
    applyDenseBoundary(bc, *dense, rhs);
    return;
  }

  throw std::runtime_error("Poisson example received unsupported matrix type");
}

} // namespace

PoissonForwardProblem::PoissonForwardProblem(const Options& opts)
  : opts_(opts),
    mesh_(makePoissonMesh(opts)),
    space_(&mesh_, &fe_),
    quad_(GaussQuadrature::make(fe_.referenceElement(), 2))
{
  space_.setup();
  pattern_ = std::make_unique<CsrPattern>(makeCsrPattern(space_));
}

const Options& PoissonForwardProblem::options() const noexcept
{
  return opts_;
}

const CsrPattern& PoissonForwardProblem::pattern() const
{
  return *pattern_;
}

Index PoissonForwardProblem::numNodes() const noexcept
{
  return mesh_.numNodes();
}

Index PoissonForwardProblem::numDofs() const noexcept
{
  return space_.numDofs();
}

void PoissonForwardProblem::assemble(AssemblyMatrix& A,
                                     Vector<Real>&   rhs) const
{
  Assembler assembler(space_);
  assembler.initMat(A);
  assembler.initVec(rhs);

  ElementValues ev(fe_, quad_);
  DenseMatrix   Ae(space_.numDofsPerElem(), space_.numDofsPerElem());

  for (Index ie = 0; ie < mesh_.numElems(); ++ie)
  {
    ev.reinit(mesh_.elem(ie));
    Ae.setZero();

    for (Index iq = 0; iq < ev.numQuadraturePoints(); ++iq)
    {
      const auto dNdx = ev.dNdx(iq);
      const Real JxW  = ev.JxW(iq);

      for (Index i = 0; i < Ae.rows(); ++i)
      {
        for (Index j = 0; j < Ae.cols(); ++j)
        {
          for (Index d = 0; d < ev.dim(); ++d)
          {
            Ae(i, j) += dNdx(i, d) * dNdx(j, d) * JxW;
          }
        }
      }
    }

    assembler.addMat(ie, Ae, A);
  }

  BoundaryCondition bc;
  bc.addBoundary(space_, onBoundary, boundaryValue);
  applyBoundary(bc, A, rhs);
}

ErrorReport PoissonForwardProblem::errorReport(const Vector<Real>& x) const
{
  if (x.size() != space_.numDofs())
  {
    throw std::runtime_error("Poisson solution vector has incompatible size");
  }

  ErrorReport report;
  report.min_value = std::numeric_limits<Real>::infinity();
  report.max_value = -std::numeric_limits<Real>::infinity();
  report.max_err   = 0.0;

  Real err2_sum = 0.0;
  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    const Real value = x[space_.globalDof(in, 0)];
    const Real err   = value - exactValue(mesh_.node(in));

    report.min_value  = std::min(report.min_value, value);
    report.max_value  = std::max(report.max_value, value);
    report.max_err    = std::max(report.max_err, std::abs(err));
    err2_sum         += err * err;
  }

  report.rms_err = std::sqrt(err2_sum / static_cast<Real>(mesh_.numNodes()));

  return report;
}

void PoissonForwardProblem::writeSolution(const Vector<Real>& x,
                                          const std::string&  output_base) const
{
  if (output_base.empty())
  {
    return;
  }
  if (x.size() != space_.numDofs())
  {
    throw std::runtime_error("Poisson solution vector has incompatible size");
  }

  const std::filesystem::path output_path = vtuPathFromBase(output_base);
  if (output_path.has_parent_path())
  {
    std::filesystem::create_directories(output_path.parent_path());
  }

  Vector<Real> solution(mesh_.numNodes());
  Vector<Real> exact(mesh_.numNodes());
  Vector<Real> error(mesh_.numNodes());
  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    solution[in] = x[space_.globalDof(in, 0)];
    exact[in]    = exactValue(mesh_.node(in));
    error[in]    = solution[in] - exact[in];
  }

  VtuWriter out;
  out.writePointData(output_path.string(),
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
                 Real                         residual_norm)
{
  const Options& opts = problem.options();
  out << "Poisson forward (" << backend << ")\n";
  out << "  cells: " << opts.num_x_cells << " x " << opts.num_y_cells
      << '\n';
  out << "  nodes: " << problem.numNodes() << '\n';
  out << "  dofs: " << problem.numDofs() << '\n';
  out << "  solution range: [" << error.min_value << ", "
      << error.max_value << "]\n";
  out << "  residual l2 norm: " << residual_norm << '\n';
  out << "  rms nodal error: " << error.rms_err << '\n';
  out << "  max nodal error: " << error.max_err << '\n';
}

} // namespace femx::examples::poisson
