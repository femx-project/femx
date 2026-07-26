#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <utility>

#include "TestHelper.hpp"
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/cuda/CudaLinearSystem.hpp>
#include <femx/linalg/native/HostLinearSystem.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/model/navier/NavierModel.hpp>
#include <femx/model/navier/NavierResidual.hpp>
#include <femx/state/TimeIntegrator.hpp>
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
  if (!linalg::CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    constexpr Index            steps = 2;
    model::navier::NavierModel model(
        fem::Mesh::makeStructuredQuad(4, 4),
        steps,
        0.1,
        {1.0, 0.1});

    const auto        vel = model.space().field(0);
    const auto        pre = model.space().field(1);
    HostVector<Index> dofs;
    HostVector<Real>  level_vals;
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

    HostVector<Real> vals(steps * dofs.size());
    for (Index step = 0; step < steps; ++step)
    {
      for (Index i = 0; i < dofs.size(); ++i)
      {
        vals[step * dofs.size() + i] = level_vals[i];
      }
    }
    const HostVector<Real> init(model.numStates());
    const HostVector<Real> prm;

    model::navier::HostNavierResidual     navier(model);
    assembly::HostConstrainedTimeResidual cpu_res(
        navier,
        fem::makeControlMap(
            steps, model.numStates(), {}, dofs, vals, {}, 0, 0));
    linalg::HostLinearSystem cpu_system(
        std::make_unique<linalg::ReSolveLinearSolver>());
    state::HostTimeIntegrator cpu(cpu_res, cpu_system);
    cpu.setInitialState(init);
    state::TimeTrajectory expected;
    cpu.solve(prm.view(), expected);

    auto control = fem::makeControlMap(
        steps, model.numStates(), {}, dofs, vals, {}, 0, 0);
    linalg::CudaLinearSystem cuda_system(
        std::make_unique<linalg::ReSolveLinearSolver>());
    auto& cuda_ctx =
        static_cast<linalg::CudaContext&>(cuda_system.context());
    model::navier::DeviceNavierResidual d_navier(
        model, cuda_ctx);
    assembly::DeviceConstrainedTimeResidual cuda_res(
        d_navier,
        std::move(control),
        {},
        cuda_ctx);
    state::DeviceTimeIntegrator cuda(cuda_res, cuda_system);
    DeviceVector<Real>          d_init;
    DeviceVector<Real>          d_prm;
    auto&                       vec_handler = cuda_ctx.vectors();
    vec_handler.copy(init, d_init);
    vec_handler.copy(prm, d_prm);
    cuda_ctx.sync();
    cuda.setInitialState(d_init);

    state::TimeTrajectory actual;
    cuda.solve(d_prm.view(), actual);
    status *= trajectoriesNear(actual, expected, 1.0e-6);
    status *= actual[1][dofs[0]] == level_vals[0];
    status *= actual[2][dofs[0]] == level_vals[0];

    const state::SolveStats repeat_stats =
        cuda.solve(d_prm.view(), actual);
    status *= repeat_stats.assm_calls == steps;
    status *= repeat_stats.lin_solve_calls == steps;
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
