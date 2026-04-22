/**
 * @file main.cpp
 * @author Kakeru Ueda (ueda.k.2290@m.isct.ac.jp)
 * @brief Solve the incompressible Navier-Stokes equations with GLS-stabilized FEM.
 *
 * GLS (Galerkin/least-squares) adds residual-based least-squares terms to the
 * Galerkin finite element formulation. This improves stability for
 * advection-dominated flow and equal-order velocity-pressure discretizations.
 *
 * @ref Tezduyar1992
 * Tezduyar, T. E., Mittal, S., Ray, S. E., and Shih, R.,
 * "Stabilized finite element formulations for incompressible flow
 * computations", Computer Methods in Applied Mechanics and Engineering, 1992.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Assembly.hpp"
#include "BoundaryConditions.hpp"
#include "Config.hpp"
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
#include <refem/solver/ReSolveLinearSolver.hpp>
#include <refem/solver/Workspace.hpp>

using namespace refem;

struct Snapshot
{
  real_type time{0.0};
  Vector    ux;
  Vector    uy;
  Vector    uz;
  Vector    p;
};

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
  throw std::runtime_error("Unsupported mesh cell type for navier-gls");
}

bool isFinite(const Vector& x)
{
  for (index_type i = 0; i < x.size(); ++i)
  {
    if (!std::isfinite(x[i]))
    {
      return false;
    }
  }
  return true;
}

void setSolverOptions(ReSolveOptions& options)
{
  options.factor             = "none";
  options.refactor           = "none";
  options.ir                 = "none";
  options.max_iterations     = 5000;
  options.restart            = 200;
  options.relative_tolerance = 1.0e-10;
  options.solve              = "fgmres";
  options.precond            = "ilu0";
  options.flexible           = true;
}

WorkspaceType workspaceType(const SolverParams& solver)
{
  if (solver.backend == "cuda")
  {
    return WorkspaceType::Cuda;
  }
  return WorkspaceType::Cpu;
}

void splitFields(const Vector&       x,
                 const BlockFESpace& space,
                 Vector&             ux,
                 Vector&             uy,
                 Vector&             uz,
                 Vector&             p)
{
  const Mesh&      mesh  = space.mesh();
  const auto       u_dof = space.field(0);
  const auto       p_dof = space.field(1);
  const index_type nd    = u_dof.numComponents();

  for (index_type node = 0; node < mesh.numNodes(); ++node)
  {
    ux[node] = x[u_dof.globalDof(node, 0)];
    uy[node] = nd > 1 ? x[u_dof.globalDof(node, 1)] : 0.0;
    uz[node] = nd > 2 ? x[u_dof.globalDof(node, 2)] : 0.0;
    p[node]  = x[p_dof.globalDof(node)];
  }
}

Snapshot makeSnapshot(const BlockFESpace& space,
                      const Vector&       x,
                      real_type           time)
{
  const index_type nodes = space.mesh().numNodes();
  Snapshot         snapshot{time, Vector(nodes), Vector(nodes), Vector(nodes), Vector(nodes)};
  splitFields(x, space, snapshot.ux, snapshot.uy, snapshot.uz, snapshot.p);
  return snapshot;
}

void writeOutput(const Mesh&                  mesh,
                 const OutputParams&          params,
                 const std::vector<Snapshot>& snapshots)
{
  TimeSeriesDataOut velocity_out;
  velocity_out.attachMesh(mesh);

  TimeSeriesDataOut pressure_out;
  pressure_out.attachMesh(mesh);

  for (const Snapshot& snapshot : snapshots)
  {
    velocity_out.beginStep(snapshot.time);
    velocity_out.addNodalVectorField("velocity",
                                     snapshot.ux,
                                     snapshot.uy,
                                     snapshot.uz);

    pressure_out.beginStep(snapshot.time);
    pressure_out.addNodalScalarField("pressure", snapshot.p);
  }

  velocity_out.write(params.directory + "/velocity");
  pressure_out.write(params.directory + "/pressure");
}

int run(const Params& params)
{
  Mesh mesh    = GmshReader::read(params.mesh_file);
  auto element = makeElement(mesh);

  FESpace velocity_space(&mesh, element.get(), mesh.dim());
  FESpace pressure_space(&mesh, element.get());

  BlockFESpace space;
  space.addField(velocity_space);
  space.addField(pressure_space);
  space.setup();

  FixedSparsityPattern pattern(space);
  SparseMatrix         A(pattern);
  Vector               b(space.numDofs());
  Vector               x(space.numDofs());
  Vector               xp(space.numDofs());
  x.setZero();
  xp.setZero();

  ReSolveOptions solver_options;
  setSolverOptions(solver_options);
  LinearSolver solver(workspaceType(params.solver),
                      SolverBackend::ReSolve,
                      solver_options);

  std::vector<Snapshot> snapshots;

  for (index_type step = 1; step <= params.time.steps; ++step)
  {
    const real_type time = step * params.time.dt;

    const auto    assembly_start = std::chrono::high_resolution_clock::now();
    AssemblyStats stats;
    assembleSystem(space,
                   x,
                   xp,
                   step == 1,
                   params.fluid,
                   params.time.dt,
                   A,
                   b,
                   stats);
    const auto assembly_end = std::chrono::high_resolution_clock::now();

    if (!std::isfinite(stats.max_cfl))
    {
      throw std::runtime_error("Stopping as CFL became invalid");
    }

    Vector x_old = x;
    auto   bc    = makeBoundaryCondition(space, params.bcs, time);
    bc.apply(A, b);
    solver.setOperator(A);

    const auto solve_start = std::chrono::high_resolution_clock::now();
    solver.solve(b, x);
    const auto solve_end = std::chrono::high_resolution_clock::now();

    if (!isFinite(x))
    {
      throw std::runtime_error("Linear solve produced non-finite values in x");
    }
    xp = x_old;

    if (step % params.output.interval == 0 || step == params.time.steps)
    {
      snapshots.push_back(makeSnapshot(space, x, time));
      writeOutput(mesh, params.output, snapshots);
    }

    const std::chrono::duration<double> assembly_elapsed =
        assembly_end - assembly_start;
    const std::chrono::duration<double> solve_elapsed =
        solve_end - solve_start;
    std::cout << "step " << std::setw(5) << step << " / "
              << params.time.steps << ", t = " << std::setw(10)
              << time << ", max CFL = " << stats.max_cfl
              << ", assembly = " << assembly_elapsed.count() << " s"
              << ", solve = " << solve_elapsed.count() << " s\n";
  }

  return 0;
}

int main(int argc, char* argv[])
{
  try
  {
    const CLOptions options = parseCommandLine(argc, argv);
    const Params    params  = loadConfig(options.config_file);
    return run(params);
  }
  catch (const std::exception& e)
  {
    std::cerr << "navier-gls: " << e.what() << '\n';
    return 1;
  }
}
