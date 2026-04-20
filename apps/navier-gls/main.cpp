#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NavierGLS.hpp"
#include "Utils.hpp"
#include <refem/bc/DirichletCondition.hpp>
#include <refem/fe/BlockFESpace.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/fe/LagrangeQuadQ1.hpp>
#include <refem/io/TimeSeriesDataOut.hpp>
#include <refem/linalg/FixedSparsityPattern.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/mesh/Mesh.hpp>
#include <refem/solver/LinearSolver.hpp>
#include <refem/solver/ReSolverLinearSolver.hpp>
#include <refem/solver/Workspace.hpp>

#ifndef REFEM_NAVIERSTOKES_OUTPUT_DIR
#define REFEM_NAVIERSTOKES_OUTPUT_DIR "."
#endif

using refem::index_type;
using refem::real_type;

int run(const Options& options)
{
  refem::Mesh           mesh = refem::Mesh::makeStructuredQuad(refem::nx, refem::ny);
  refem::LagrangeQuadQ1 element;
  refem::FESpace        velocity_space(&mesh, &element, refem::dim);
  refem::FESpace        pressure_space(&mesh, &element);
  refem::BlockFESpace   space;
  space.addField(velocity_space);
  space.addField(pressure_space);
  space.setup();

  refem::FixedSparsityPattern pattern(space);
  refem::SparseMatrix         A(pattern);

  refem::Vector b(space.numDofs());
  refem::Vector x(space.numDofs());
  refem::Vector xp(space.numDofs());
  x.setZero();
  xp.setZero();

  refem::SolverBackend  solver_backend = refem::SolverBackend::ReSolve;
  refem::ReSolveOptions solver_options;
  setSolverOptions(solver_options);

  refem::WorkspaceType workspace;
  if (options.backend == "cpu")
  {
    workspace = refem::WorkspaceType::Cpu;
  }
  else if (options.backend == "cuda")
  {
    workspace = refem::WorkspaceType::Cuda;
  }

  refem::LinearSolver solver(workspace, solver_backend, solver_options);
  const auto          bc = refem::cavityBoundary(space);

  std::vector<refem::Snapshot> snapshots;

  for (index_type step = 1; step <= refem::steps; ++step)
  {
    const auto start = std::chrono::high_resolution_clock::now();

    real_type max_cfl = 0.0;
    refem::assembleSystem(space,
                          x,
                          xp,
                          step == 1,
                          A,
                          b,
                          max_cfl);

    if (!std::isfinite(max_cfl) || max_cfl > refem::max_cfl)
    {
      throw std::runtime_error(
          "Stopping as CFL became invalid or too large. ");
    }

    refem::Vector x_old = x;
    bc.apply(A, b);

    solver.setOperator(A);
    solver.solve(b, x);

    if (!isFinite(x))
    {
      throw std::runtime_error(
          "Linear solve produced non-finite values in x");
    }

    xp = x_old;

    if (step % refem::interval == 0 || step == refem::steps)
    {
      snapshots.push_back(refem::makeSnapshot(space,
                                              x,
                                              step * refem::dt));
    }

    const auto                          end     = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cout << "step " << std::setw(5) << step << " / "
              << refem::steps << ", t = " << std::setw(10)
              << step * refem::dt
              << ", max CFL = " << max_cfl
              << ", wall = " << elapsed.count() << " s\n";
  }

  writeTimeSeriesOutput(mesh, snapshots);

  return 0;
}

int main(int argc, char* argv[])
{
  try
  {
    const Options options = parseOptions(argc, argv);
    return run(options);
  }
  catch (const std::exception& e)
  {
    std::cerr << "navierstokes: " << e.what() << '\n';
    return 1;
  }
}
