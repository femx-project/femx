#include <cmath>
#include <exception>
#include <iostream>

#include "TestHelper.hpp"
#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/model/ns/NavierStokesModel.hpp>
#include <femx/model/ns/ResolveTimeIntegrator.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx::tests
{
namespace
{

bool trajectoriesNear(const state::TimeTrajectory& lhs,
                      const state::TimeTrajectory& rhs,
                      Real                         tol)
{
  if (lhs.numSteps() != rhs.numSteps()
      || lhs.numStates() != rhs.numStates())
  {
    return false;
  }
  for (Index i = 0; i < lhs.size(); ++i)
  {
    if (std::abs(lhs.data()[i] - rhs.data()[i]) > tol)
    {
      return false;
    }
  }
  return true;
}

TestOutcome resolveCudaAdvancesTwoSteps()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    constexpr Index              steps = 2;
    model::ns::NavierStokesModel model(
        fem::Mesh::makeStructuredQuad(4, 4),
        steps,
        0.1,
        {1.0, 0.1});

    const auto   vel = model.space().field(0);
    const auto   pre = model.space().field(1);
    Array<Index> dofs;
    HostVector   level_vals;
    for (Index in = 0; in < model.mesh().numNodes(); ++in)
    {
      const auto& point = model.mesh().node(in);
      if (point[0] != 0.0 && point[0] != 1.0
          && point[1] != 0.0 && point[1] != 1.0)
      {
        continue;
      }
      dofs.push_back(vel.globalDof(in, 0));
      level_vals.push_back(point[1] == 1.0 ? 1.0 : 0.0);
      dofs.push_back(vel.globalDof(in, 1));
      level_vals.push_back(0.0);
    }
    dofs.push_back(pre.globalDof(0));
    level_vals.push_back(0.0);

    HostVector vals(steps * dofs.size());
    for (Index step = 0; step < steps; ++step)
    {
      for (Index i = 0; i < dofs.size(); ++i)
      {
        vals[step * dofs.size() + i] = level_vals[i];
      }
    }
    const HostVector init(model.numStates());
    const HostVector prm;

    assembly::TimeDirichletControlResidual cpu_res(
        model.residual(), fem::DirichletControl{}, dofs, 0, 0, vals);
    linalg::MapCsrMatrix        cpu_mat(model.map());
    linalg::ReSolveLinearSolver cpu_solver;
    state::TimeLinearIntegrator cpu(cpu_res, cpu_mat, cpu_solver);
    cpu.setInitialState(init);
    state::TimeTrajectory expected;
    cpu.solve(prm, expected);

    model::ns::ResolveTimeIntegrator cuda(model, dofs, vals);
    cuda.setInitialState(init);
    state::TimeTrajectory actual;
    cuda.solve(prm, actual);
    status *= trajectoriesNear(actual, expected, 1.0e-6);
    status *= actual[1][dofs[0]] == level_vals[0];
    status *= actual[2][dofs[0]] == level_vals[0];

    cuda.solve(prm, actual);
    status *= cuda.assemblyCalls() == 2 * steps;
    status *= cuda.solveCalls() == 2 * steps;
    status *= trajectoriesNear(actual, expected, 1.0e-6);
  }
  catch (const std::exception& error)
  {
    std::cout << "    exception: " << error.what() << '\n';
    status *= false;
  }

  return status.report();
}

} // namespace
} // namespace femx::tests

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::resolveCudaAdvancesTwoSteps();
  return results.summary();
}
