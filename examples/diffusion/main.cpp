#include <chrono>
#include <iostream>
#include <memory>
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
#include <refem/solver/ReSolverLinearSolver.hpp>
#include <resolve/resolve_defs.hpp>
#include <resolve/utilities/params/CliOptions.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif

#ifndef REFEM_DIFFUSION_OUTPUT_DIR
#define REFEM_DIFFUSION_OUTPUT_DIR "."
#endif

using namespace refem;

void printHelpInfo()
{
  std::cout << "\ndiffusion solves a left-to-right diffusion problem.\n\n";
  std::cout << "Usage:\n\t./diffusion [-b <cpu|cuda>]\n\n";
  std::cout << "Optional features:\n";
  std::cout << "\t-b <cpu|cuda> \tSelects hardware backend.\n";
  std::cout << "\t-h \tPrints this message.\n\n";
}

template <class workspace_type>
int diffusion(const std::string& backend)
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

  workspace_type workspace;
  workspace.initializeHandles();

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

  ReSolveLinearSolver solver(&workspace, options);
  solver.setOperator(form.matrix());

  const auto start = std::chrono::high_resolution_clock::now();
  solver.solve(b, x);
  const auto                          end  = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time = end - start;

  DataOut data_out;
  data_out.attachMesh(mesh);
  data_out.addNodalField("u", x);

  const std::string filename =
      std::string(REFEM_DIFFUSION_OUTPUT_DIR) + "/diffusion";
  data_out.write(filename);

  std::cout << "Backend: " << backend << '\n';
  std::cout << "Solve time: " << time.count() << " s\n";

  return 0;
}

int main(int argc, char* argv[])
{
  ReSolve::CliOptions options(argc, argv);

  if (options.hasKey("-h"))
  {
    printHelpInfo();
    return 0;
  }

  auto opt = options.getParamFromKey("-b");
  if (!opt)
  {
    std::cout << "No backend option provided. Defaulting to CPU.\n";
    return diffusion<ReSolve::LinAlgWorkspaceCpu>("cpu");
  }
#ifdef RESOLVE_USE_CUDA
  else if (opt->second == "cuda")
  {
    return diffusion<ReSolve::LinAlgWorkspaceCUDA>("cuda");
  }
#endif
  else if (opt->second == "cpu")
  {
    return diffusion<ReSolve::LinAlgWorkspaceCpu>("cpu");
  }
  else
  {
    std::cout << "Re::Solve is not built with support for " << opt->second
              << " backend.\n";
    return 1;
  }

  return 0;
}
