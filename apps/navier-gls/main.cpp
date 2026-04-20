#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "NavierGLS.hpp"
#include "Utils.hpp"
#include <refem/bc/DirichletCondition.hpp>
#include <refem/fe/BlockFESpace.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/fe/LagrangeQuadQ1.hpp>
#include <refem/fe/LagrangeTetrahedronP1.hpp>
#include <refem/fe/LagrangeTriangleP1.hpp>
#include <refem/io/TimeSeriesDataOut.hpp>
#include <refem/linalg/FixedSparsityPattern.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/mesh/GmshReader.hpp>
#include <refem/mesh/Mesh.hpp>
#include <refem/solver/LinearSolver.hpp>
#include <refem/solver/ReSolverLinearSolver.hpp>
#include <refem/solver/Workspace.hpp>

#ifndef REFEM_NAVIERSTOKES_OUTPUT_DIR
#define REFEM_NAVIERSTOKES_OUTPUT_DIR "."
#endif

using namespace refem;

std::unique_ptr<FiniteElement> makeElement(const Mesh& mesh)
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
  throw std::runtime_error("Unsupported mesh cell type for navier_gls");
}

int run(const Options& options)
{
  Mesh         mesh    = GmshReader::read(options.mesh_file);
  auto         element = makeElement(mesh);
  FESpace      velocity_space(&mesh, element.get(), mesh.dim());
  FESpace      pressure_space(&mesh, element.get());
  BlockFESpace space;

  space.addField(velocity_space);
  space.addField(pressure_space);
  space.setup();

  FixedSparsityPattern pattern(space);
  SparseMatrix         A(pattern);

  Vector b(space.numDofs());
  Vector x(space.numDofs());
  Vector xp(space.numDofs());
  x.setZero();
  xp.setZero();

  SolverBackend  solver_backend = SolverBackend::ReSolve;
  ReSolveOptions solver_options;
  setSolverOptions(solver_options);

  WorkspaceType workspace;
  if (options.backend == "cpu")
  {
    workspace = WorkspaceType::Cpu;
  }
  else if (options.backend == "cuda")
  {
    workspace = WorkspaceType::Cuda;
  }

  LinearSolver solver(workspace, solver_backend, solver_options);
  const auto   bc = navierBoundary(space);

  std::vector<Snapshot> snapshots;

  for (index_type step = 1; step <= steps; ++step)
  {
    real_type  step_max_cfl   = 0.0;
    const auto assembly_start = std::chrono::high_resolution_clock::now();
    assembleSystem(space,
                   x,
                   xp,
                   step == 1,
                   A,
                   b,
                   step_max_cfl);
    const auto assembly_end = std::chrono::high_resolution_clock::now();

    if (!std::isfinite(step_max_cfl))
    {
      throw std::runtime_error("Stopping as CFL became invalid.");
    }

    Vector x_old = x;

    bc.apply(A, b);
    solver.setOperator(A);

    const auto solve_start = std::chrono::high_resolution_clock::now();
    solver.solve(b, x);
    const auto solve_end = std::chrono::high_resolution_clock::now();

    if (!isFinite(x))
    {
      throw std::runtime_error(
          "Linear solve produced non-finite values in x");
    }
    xp = x_old;

    double snapshot_seconds = 0.0;
    if (step % interval == 0 || step == steps)
    {
      snapshots.push_back(makeSnapshot(space,
                                       x,
                                       step * dt));
    }

    const auto step_end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> assembly_elapsed =
        assembly_end - assembly_start;
    const std::chrono::duration<double> solve_elapsed =
        solve_end - solve_start;
    std::cout << "step " << std::setw(5) << step << " / "
              << steps << ", t = " << std::setw(10)
              << step * dt
              << ", max CFL = " << step_max_cfl
              << ", assembly = " << assembly_elapsed.count() << " s"
              << ", solve = " << solve_elapsed.count() << " s\n";
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
