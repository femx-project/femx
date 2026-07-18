#include <cmath>
#include <exception>
#include <iostream>

#include "TestHelper.hpp"
#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/inverse/SumTimeObjective.hpp>
#include <femx/inverse/TimeLeastSquaresObjective.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/inverse/TimeRegularization.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/model/ns/NavierStokesModel.hpp>
#include <femx/model/ns/ResolveTimeIntegrator.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>

namespace femx::tests
{
namespace
{

bool near(Real lhs, Real rhs, Real tol)
{
  return std::abs(lhs - rhs)
         <= tol * (1.0 + std::max(std::abs(lhs), std::abs(rhs)));
}

bool vectorsNear(const HostVector& lhs,
                 const HostVector& rhs,
                 Real              tol)
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

class InitialMapIntegrator final : public state::TimeIntegrator
{
public:
  InitialMapIntegrator(state::TimeLinearIntegrator&    integ,
                       const fem::HostInitialStateMap& init)
    : integ_(integ), init_(init)
  {
  }

  Index numSteps() const override
  {
    return integ_.numSteps();
  }

  Index numStates() const override
  {
    return init_.numStates();
  }

  Index numParams() const override
  {
    return init_.numParams();
  }

  void solve(const HostVector&      prm,
             state::TimeTrajectory& tr) override
  {
    HostVector state(init_.numStates());
    fem::initialState(init_.view(), prm.view(), state.view());
    integ_.setInitialState(state);
    integ_.solve(prm, tr);
  }

private:
  state::TimeLinearIntegrator&    integ_;
  const fem::HostInitialStateMap& init_;
};

class InitialGradMap final : public inverse::InitialStateGradientMap
{
public:
  explicit InitialGradMap(const fem::HostInitialStateMap& init)
    : init_(init)
  {
  }

  void apply(const HostVector&,
             const HostVector& state_grad,
             HostVector&       out) override
  {
    out.resize(init_.numParams());
    out.setZero();
    fem::addInitialJacT(init_.view(), state_grad.view(), out.view());
  }

private:
  const fem::HostInitialStateMap& init_;
};

struct ProblemData
{
  fem::DirichletControl      ctr;
  Array<Index>               fixed_dofs;
  HostVector                 fixed_vals;
  Array<LinearInterpolation> time;
  Index                      init_dof{-1};
};

ProblemData makeProblemData(const model::ns::NavierStokesModel& model)
{
  const auto vel = model.space().field(0);
  const auto pre = model.space().field(1);

  Index        ctr_dof  = -1;
  Index        init_dof = -1;
  Array<Index> fixed;
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

  HostVector                 vals(model.numSteps() * fixed.size());
  Array<LinearInterpolation> time(model.numSteps());
  for (Index step = 0; step < model.numSteps(); ++step)
  {
    time[step] = {step, step, 0.0};
  }
  return {fem::DirichletControl(Array<Index>{ctr_dof}),
          std::move(fixed),
          std::move(vals),
          std::move(time),
          init_dof};
}

TestOutcome resolveCudaReducedGradientMatchesCpuAndFd()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    constexpr Index              steps = 3;
    model::ns::NavierStokesModel model(
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
    HostVector  mean(model.numStates());
    DenseMatrix modes(model.numStates(), 1);
    modes(data.init_dof, 0)       = 0.2;
    fem::HostInitialStateMap init = fem::makeInitialStateMap(
        mean, modes, data.ctr, 0, 1, num_prm);

    fem::TimePointInterpolator obs(
        steps,
        model.space(),
        0,
        Array<Point3>{{0.5, 0.5, 0.0}},
        Array<Index>{0},
        num_prm);
    inverse::TimeObservationData obs_data(steps + 1, 1);
    obs_data.setZero();
    inverse::TimeLeastSquaresObjective misfit(obs, obs_data, 1.0);
    inverse::TimeRegularization        reg(
        steps, model.numStates(), 3, 1, 0.05, 0.02);
    inverse::SumTimeObjective obj(
        steps, model.numStates(), num_prm);
    obj.add(misfit).add(reg);

    assembly::TimeDirichletControlResidual cpu_res(
        model.residual(),
        data.ctr,
        data.fixed_dofs,
        1,
        num_prm,
        data.fixed_vals,
        data.time);
    linalg::MapCsrMatrix           fwd_mat(model.map());
    linalg::ReSolveLinearSolver    fwd_solver;
    state::TimeLinearIntegrator    base(cpu_res, fwd_mat, fwd_solver);
    InitialMapIntegrator           cpu_integ(base, init);
    state::TimeLinearization       lin;
    linalg::MapCsrMatrix           next(model.map());
    linalg::MapCsrMatrix           hist(model.map());
    linalg::ReSolveLinearSolver    adj_solver;
    inverse::TimeReducedFunctional cpu(
        cpu_integ, cpu_res, lin, next, hist, adj_solver, obj);
    InitialGradMap init_grad(init);
    cpu.setInitialStateParamJacT(&init_grad);

    model::ns::ResolveTimeIntegrator cuda_integ(
        model, ctr, init);
    model::ns::ResolveTimeReducedFunctional cuda(cuda_integ, obj);

    const HostVector prm{0.25, 0.35, 0.65};
    HostVector       cpu_grad;
    HostVector       cuda_grad;
    const Real       cpu_val   = cpu.valueGrad(prm, cpu_grad);
    const Real       cuda_val  = cuda.valueGrad(prm, cuda_grad);
    status                    *= near(cuda_val, cpu_val, 2.0e-6);
    status                    *= vectorsNear(cuda_grad, cpu_grad, 2.0e-5);

    HostVector repeat_grad;
    const Real repeat_val  = cuda.valueGrad(prm, repeat_grad);
    status                *= near(repeat_val, cuda_val, 1.0e-12);
    status                *= vectorsNear(repeat_grad, cuda_grad, 1.0e-12);

    HostVector     fd(num_prm);
    constexpr Real eps = 1.0e-6;
    for (Index i = 0; i < num_prm; ++i)
    {
      HostVector plus   = prm;
      HostVector minus  = prm;
      plus[i]          += eps;
      minus[i]         -= eps;
      fd[i]             = (cuda.value(plus) - cuda.value(minus)) / (2.0 * eps);
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
  results +=
      femx::tests::resolveCudaReducedGradientMatchesCpuAndFd();
  return results.summary();
}
