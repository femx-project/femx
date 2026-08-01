#include <cmath>
#include <exception>
#include <iostream>
#include <memory>

#include "TestHelper.hpp"
#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/inverse/SumTimeObjective.hpp>
#include <femx/inverse/TimeLeastSquaresObjective.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/inverse/TimeRegularization.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaLinearSystem.hpp>
#include <femx/linalg/host/HostContext.hpp>
#include <femx/linalg/host/HostLinearSystem.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/model/navier/NavierModel.hpp>
#include <femx/model/navier/NavierResidual.hpp>
#include <femx/state/TimeIntegrator.hpp>

namespace femx::tests
{
namespace
{

bool near(Real lhs, Real rhs, Real tol)
{
  return std::abs(lhs - rhs)
         <= tol * (1.0 + std::max(std::abs(lhs), std::abs(rhs)));
}

bool vectorsNear(const HostVector<Real>& lhs,
                 const HostVector<Real>& rhs,
                 Real                    tol)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }
  for (Index i = 0; i < lhs.size(); ++i)
  {
    if (!near(lhs[i], rhs[i], tol))
    {
      return false;
    }
  }
  return true;
}

Real deviceValue(inverse::DeviceTimeReducedFunctional& functional,
                 const HostVector<Real>&               prm)
{
  return functional.value(prm.view());
}

Real deviceValueGrad(inverse::DeviceTimeReducedFunctional& functional,
                     const HostVector<Real>&               prm,
                     HostVector<Real>&                     grad)
{
  grad.resize(functional.numParams());
  return functional.valueGrad(prm.view(), grad.view());
}

struct ProblemData
{
  fem::DirichletControl           ctr;
  HostVector<Index>               fixed_dofs;
  HostVector<Real>                fixed_vals;
  HostVector<LinearInterpolation> time;
  Index                           init_dof{-1};
};

ProblemData makeProblemData(const model::navier::NavierModel& model)
{
  const auto vel = model.space().field(0);
  const auto pre = model.space().field(1);

  Index             ctr_dof  = -1;
  Index             init_dof = -1;
  HostVector<Index> fixed;
  for (Index in = 0; in < model.mesh().numNodes(); ++in)
  {
    const auto& pt = model.mesh().node(in);
    if (pt[0] != 0.0 && pt[0] != 1.0
        && pt[1] != 0.0 && pt[1] != 1.0)
    {
      if (init_dof < 0)
      {
        init_dof = vel.globalDof(in, 0);
      }
      continue;
    }

    const Index ux = vel.globalDof(in, 0);
    if (ctr_dof < 0 && pt[1] == 1.0
        && pt[0] > 0.0 && pt[0] < 1.0)
    {
      ctr_dof = ux;
    }
    else
    {
      fixed.push_back(ux);
    }
    fixed.push_back(vel.globalDof(in, 1));
  }
  fixed.push_back(pre.globalDof(0));

  HostVector<Real>                vals(model.numSteps() * fixed.size());
  HostVector<LinearInterpolation> time(model.numSteps());
  for (Index step = 0; step < model.numSteps(); ++step)
  {
    time[step] = {step, step, 0.0};
  }
  return {fem::DirichletControl(HostVector<Index>{ctr_dof}),
          std::move(fixed),
          std::move(vals),
          std::move(time),
          init_dof};
}

TestOutcome resolveCudaHistoryVjpMatchesCpu()
{
  TestStatus status(__func__);
  if (!linalg::CudaContext::available() || !ad::has_enzyme)
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
    model::navier::HostNavierResidual host_res(model);
    linalg::HostContext               host_ctx;
    linalg::CudaContext               cuda_ctx;
    model::navier::CudaNavierResidual device_res(model, cuda_ctx);

    const Index      num_states = model.numStates();
    HostVector<Real> hist(2 * num_states);
    HostVector<Real> nxt(num_states);
    HostVector<Real> adj(num_states);
    HostVector<Real> prm;
    for (Index i = 0; i < hist.size(); ++i)
    {
      hist[i] = 0.01 * std::sin(0.17 * (i + 1));
    }
    for (Index i = 0; i < num_states; ++i)
    {
      nxt[i] = 0.02 * std::cos(0.11 * (i + 1));
      adj[i] = 0.03 * std::sin(0.13 * (i + 1));
    }

    DeviceVector<Real> d_hist;
    DeviceVector<Real> d_nxt;
    DeviceVector<Real> d_adj;
    DeviceVector<Real> d_out;
    auto&              vec_handler = cuda_ctx.vectorHandler();
    vec_handler.copy(hist, d_hist);
    vec_handler.copy(nxt, d_nxt);
    vec_handler.copy(adj, d_adj);

    for (Index step = 0; step < steps; ++step)
    {
      for (Index lag = 0; lag < 2; ++lag)
      {
        const state::HostTimeContext host_time{
            step,
            nxt.view(),
            prm.view(),
            {hist.data(), 2, num_states}};
        HostVector<Real> expected;
        host_res.applyJacT(host_time,
                           state::VariableBlock::hist(lag),
                           adj.view(),
                           expected,
                           host_ctx);

        const state::DeviceTimeContext device_time{
            step,
            d_nxt.view(),
            {},
            {d_hist.data(), 2, num_states}};
        device_res.applyJacT(device_time,
                             state::VariableBlock::hist(lag),
                             d_adj.view(),
                             d_out,
                             cuda_ctx);

        HostVector<Real> actual;
        vec_handler.copy(d_out, actual);
        cuda_ctx.sync();
        status *= vectorsNear(actual, expected, 1.0e-11);
      }
    }
  }
  catch (const std::exception& error)
  {
    std::cout << "    exception: " << error.what() << '\n';
    status *= false;
  }

  return status.report();
}

TestOutcome resolveCudaReducedGradientMatchesCpuAndFd()
{
  TestStatus status(__func__);
  if (!linalg::CudaContext::available() || !ad::has_enzyme)
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    constexpr Index            steps = 3;
    model::navier::NavierModel model(
        fem::Mesh::makeStructuredQuad(4, 4),
        steps,
        0.1,
        {1.0, 0.1});
    ProblemData data = makeProblemData(model);
    data.time[0]     = {0, 0, 0.0};
    data.time[1]     = {0, 1, 0.5};
    data.time[2]     = {0, 0, 0.0};

    constexpr Index     num_prm = 3;
    fem::HostControlMap ctr     = fem::makeControlMap(
        steps,
        model.numStates(),
        data.ctr,
        data.fixed_dofs,
        data.fixed_vals,
        data.time,
        1,
        num_prm);
    HostVector<Real> mean(model.numStates());
    DenseMatrix      modes(model.numStates(), 1);
    modes(data.init_dof, 0)       = 0.2;
    fem::HostInitialStateMap init = fem::makeInitialStateMap(
        mean, modes, data.ctr, 0, 1, num_prm);

    fem::TimePointInterpolator obs(
        steps,
        model.space(),
        0,
        HostVector<Point3>{{0.5, 0.5, 0.0}},
        HostVector<Index>{0},
        num_prm);
    inverse::TimeObservationData obs_data(steps + 1, 1);
    obs_data.setZero();
    inverse::TimeLeastSquaresObjective misfit(obs, obs_data, 1.0);
    inverse::TimeRegularization        reg(
        steps, model.numStates(), 3, 1, 0.05, 0.02);
    inverse::SumTimeObjective obj(
        steps, model.numStates(), num_prm);
    obj.add(misfit).add(reg);

    model::navier::HostNavierResidual     navier(model);
    assembly::HostConstrainedTimeResidual cpu_res(
        navier, ctr, init);
    linalg::HostLinearSystem cpu_fwd_system(
        std::make_unique<linalg::ReSolveLinearSolver>());
    state::HostTimeIntegrator cpu_integ(cpu_res, cpu_fwd_system);
    linalg::HostLinearSystem  cpu_adj_system(
        std::make_unique<linalg::ReSolveLinearSolver>());
    inverse::HostTimeReducedFunctional cpu(
        cpu_integ, cpu_adj_system, obj);

    linalg::CudaLinearSystem cuda_fwd_system(
        std::make_unique<linalg::ReSolveLinearSolver>());
    auto& cuda_ctx =
        static_cast<linalg::CudaContext&>(cuda_fwd_system.context());
    model::navier::CudaNavierResidual d_navier(
        model, cuda_ctx);
    assembly::DeviceConstrainedTimeResidual cuda_res(
        d_navier,
        ctr,
        init,
        cuda_ctx);
    state::DeviceTimeIntegrator cuda_integ(cuda_res, cuda_fwd_system);
    linalg::CudaLinearSystem    cuda_adj_system(
        std::make_unique<linalg::ReSolveLinearSolver>());
    inverse::DeviceTimeReducedFunctional cuda(
        cuda_integ, cuda_adj_system, obj);

    const HostVector<Real> prm{0.25, 0.35, 0.65};
    HostVector<Real>       cpu_grad(num_prm);
    HostVector<Real>       cuda_grad;
    const Real             cpu_val   = cpu.valueGrad(prm.view(), cpu_grad.view());
    const Real             cuda_val  = deviceValueGrad(cuda, prm, cuda_grad);
    status                          *= near(cuda_val, cpu_val, 2.0e-6);
    status                          *= vectorsNear(cuda_grad, cpu_grad, 2.0e-5);
    status                          *= cpu.assemblyCalls() == 2 * steps;
    status                          *= cpu.solveCalls() == 2 * steps;
    status                          *= cuda.assemblyCalls() == 2 * steps;
    status                          *= cuda.solveCalls() == 2 * steps;

    HostVector<Real> repeat_grad;
    const Real       repeat_val  = deviceValueGrad(cuda, prm, repeat_grad);
    status                      *= near(repeat_val, cuda_val, 1.0e-12);
    status                      *= vectorsNear(repeat_grad, cuda_grad, 1.0e-12);

    HostVector<Real> fd(num_prm);
    constexpr Real   eps = 1.0e-6;
    for (Index i = 0; i < num_prm; ++i)
    {
      HostVector<Real> plus   = prm;
      HostVector<Real> minus  = prm;
      plus[i]                += eps;
      minus[i]               -= eps;
      fd[i]                   = (deviceValue(cuda, plus)
               - deviceValue(cuda, minus))
              / (2.0 * eps);
    }
    status *= vectorsNear(cuda_grad, fd, 3.0e-5);
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
  results += femx::tests::resolveCudaHistoryVjpMatchesCpu();
  results +=
      femx::tests::resolveCudaReducedGradientMatchesCpuAndFd();
  return results.summary();
}
