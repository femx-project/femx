#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <refem/bc/DirichletCondition.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/fe/LagrangeQuadQ1.hpp>
#include <refem/forms/BilinearForm.hpp>
#include <refem/forms/integrators/DiffusionIntegrator.hpp>
#include <refem/io/DataOut.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/linalg/Vector.hpp>
#include <refem/mesh/Mesh.hpp>
#include <refem/solver/LinearSolver.hpp>
#include <refem/solver/ReSolverLinearSolver.hpp>
#include <refem/solver/Workspace.hpp>

#ifndef REFEM_DIFFUSION_OUTPUT_DIR
#define REFEM_DIFFUSION_OUTPUT_DIR "."
#endif

using namespace refem;

struct Options
{
  WorkspaceType workspace_type = WorkspaceType::Cpu;
  std::string   backend        = "cpu";
};

void printHelpInfo()
{
  std::cout << "\ndiffusion solves a left-to-right diffusion problem.\n\n";
  std::cout << "Usage:\n\t./diffusion [-b|--backend <cpu|cuda>]\n\n";
  std::cout << "Optional features:\n";
  std::cout << "\t-b, --backend <cpu|cuda> \tSelects hardware backend.\n";
  std::cout << "\t-h, --help \tPrints this message.\n\n";
}

Options parseOptions(int argc, char* argv[])
{
  Options options;

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
      printHelpInfo();
      std::exit(0);
    }
    else if (key == "-b" || key == "--backend")
    {
      options.backend = requireValue(i, key);
    }
    else
    {
      throw std::runtime_error("Unknown option: " + key);
    }
  }

  if (options.backend == "cpu")
  {
    options.workspace_type = WorkspaceType::Cpu;
  }
  else if (options.backend == "cuda")
  {
    options.workspace_type = WorkspaceType::Cuda;
  }
  else
  {
    throw std::runtime_error(
        "Backend must be either 'cpu' or 'cuda'");
  }

  return options;
}

int run(WorkspaceType workspace_type, const std::string& backend)
{
  constexpr index_type nx = 64;
  constexpr index_type ny = 64;

  Mesh mesh = Mesh::makeStructuredQuad(nx, ny);

  LagrangeQuadQ1 element;
  FESpace        space(&mesh, &element);
  space.setup();

  BilinearForm form(&space);
  form.addDomainIntegrator(
      std::make_unique<DiffusionIntegrator>(1.0));
  form.assemble();

  Vector b(space.numDofs());
  Vector x(space.numDofs());

  DirichletCondition boundary;
  for (index_type j = 0; j <= ny; ++j)
  {
    const index_type left  = j * (nx + 1);
    const index_type right = left + nx;

    boundary.addDof(left, 1.0);
    boundary.addDof(right, 0.0);
  }
  boundary.apply(form.matrix(), b);

  ReSolveOptions options;
  options.factor             = "none";
  options.refactor           = "none";
  options.solve              = "randgmres";
  options.precond            = "ilu0";
  options.ir                 = "none";
  options.gram_schmidt       = "cgs2";
  options.sketching          = "count";
  options.flexible           = true;
  options.max_iterations     = 2500;
  options.restart            = 200;
  options.relative_tolerance = 1.0e-12;

  LinearSolver solver(workspace_type, SolverBackend::ReSolve, options);
  solver.setOperator(form.matrix());

  const auto start = std::chrono::high_resolution_clock::now();
  solver.solve(b, x);
  const auto                          end  = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time = end - start;

  DataOut dout;
  dout.attachMesh(mesh);
  dout.addNodalField("u", x);

  const std::string filename =
      std::string(REFEM_DIFFUSION_OUTPUT_DIR) + "/diffusion";
  dout.write(filename);

  std::cout << "Backend: " << backend << '\n';
  std::cout << "Solve time: " << time.count() << " s\n";

  return 0;
}

int main(int argc, char* argv[])
{
  try
  {
    const Options options = parseOptions(argc, argv);
    return run(options.workspace_type, options.backend);
  }
  catch (const std::exception& e)
  {
    std::cerr << "diffusion: " << e.what() << '\n';
    return 1;
  }
}
