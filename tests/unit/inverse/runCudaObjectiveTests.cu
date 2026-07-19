#include <cuda_runtime_api.h>

#include <cmath>
#include <exception>
#include <iostream>

#include "TestHelper.hpp"
#include <femx/inverse/SumTimeObjective.hpp>
#include <femx/inverse/TimeBlockRegularization.hpp>
#include <femx/inverse/TimeLeastSquaresObjective.hpp>
#include <femx/inverse/TimeObjectivePlan.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeRegularization.hpp>

namespace femx
{
namespace tests
{
namespace
{

__global__ void observeKernel(const Real* state, Real* out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    out[0] = state[0] + 2.0 * state[1];
    out[1] = -state[1] + 0.5 * state[2];
  }
}

__global__ void addStateJacTKernel(const Real* dir, Real* out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    out[0] += dir[0];
    out[1] += 2.0 * dir[0] - dir[1];
    out[2] += 0.5 * dir[1];
  }
}

class LinearObservation final
  : public inverse::TimeObservationOperator,
    public inverse::DeviceTimeObservationOperator
{
public:
  Index numSteps() const override
  {
    return 2;
  }

  Index numStates() const override
  {
    return 3;
  }

  Index numParams() const override
  {
    return 4;
  }

  Index numObservations() const override
  {
    return 2;
  }

  std::unique_ptr<inverse::DeviceTimeObservationOperator> copyToDevice(
      CudaContext&) const override
  {
    return std::make_unique<LinearObservation>();
  }

  void observe(Index,
               const HostVector& state,
               const HostVector&,
               HostVector& out) const override
  {
    out = {state[0] + 2.0 * state[1],
           -state[1] + 0.5 * state[2]};
  }

  void applyStateJac(Index,
                     const HostVector&,
                     const HostVector&,
                     const HostVector& dir,
                     HostVector&       out) const override
  {
    out = {dir[0] + 2.0 * dir[1],
           -dir[1] + 0.5 * dir[2]};
  }

  void applyStateJacT(Index,
                      const HostVector&,
                      const HostVector&,
                      const HostVector& dir,
                      HostVector&       out) const override
  {
    out = {dir[0], 2.0 * dir[0] - dir[1], 0.5 * dir[1]};
  }

  void applyParamJac(Index,
                     const HostVector&,
                     const HostVector&,
                     const HostVector&,
                     HostVector& out) const override
  {
    out.resize(numObservations());
  }

  void applyParamJacT(Index,
                      const HostVector&,
                      const HostVector&,
                      const HostVector&,
                      HostVector& out) const override
  {
    out.resize(numParams());
  }

  void observe(Index,
               DeviceConstVectorView state,
               DeviceVectorView      out,
               CudaContext&          ctx) const override
  {
    observeKernel<<<1,
                    1,
                    0,
                    static_cast<cudaStream_t>(ctx.stream())>>>(
        state.data(), out.data());
    device::checkLastError();
  }

  void addStateJacT(Index,
                    DeviceConstVectorView dir,
                    DeviceVectorView      out,
                    CudaContext&          ctx) const override
  {
    addStateJacTKernel<<<1,
                         1,
                         0,
                         static_cast<cudaStream_t>(ctx.stream())>>>(
        dir.data(), out.data());
    device::checkLastError();
  }
};

bool near(Real a, Real b, Real tol = 1.0e-11)
{
  return std::abs(a - b) <= tol;
}

bool near(const HostVector& a,
          const HostVector& b,
          Real              tol = 1.0e-11)
{
  if (a.size() != b.size())
  {
    return false;
  }
  for (Index i = 0; i < a.size(); ++i)
  {
    if (!near(a[i], b[i], tol))
    {
      return false;
    }
  }
  return true;
}

Real inner(const HostVector& a, const HostVector& b)
{
  Real val = 0.0;
  for (Index i = 0; i < a.size(); ++i)
  {
    val += a[i] * b[i];
  }
  return val;
}

TestOutcome deviceObjectiveMatchesHostAndFiniteDifference()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    LinearObservation            obs;
    inverse::TimeObservationData data(2, 2);
    data[0] = HostVector{0.7, -0.2};
    data[1] = HostVector{1.1, 0.4};
    data.setTimeValues({0.5, 1.5});

    inverse::TimeLeastSquaresObjective ls(
        obs,
        data,
        {1.0, 1.5, 0.75},
        {1.0, 0.5, 2.0, 1.25},
        1.0);
    inverse::TimeRegularization time_reg(
        2,
        3,
        2,
        2,
        0.7,
        0.2,
        {0.1, -0.2, 0.3, 0.4});
    inverse::TimeBlockRegularization block_reg(
        2,
        3,
        1,
        4,
        {0, 0, 1, 2, 3},
        {0, 1, 2, 3, 1},
        {2.0, -0.5, 1.25, 0.75, -0.2},
        0.4,
        {-0.1, 0.2, 0.0, 0.3});

    inverse::SumTimeObjective regs(2, 3, 4);
    regs.add(time_reg).add(block_reg);
    inverse::SumTimeObjective obj(2, 3, 4);
    obj.add(ls).add(regs);

    state::TimeTrajectory tr(2, 3);
    tr[0] = HostVector{0.2, -0.4, 0.7};
    tr[1] = HostVector{1.0, 0.3, -0.2};
    tr[2] = HostVector{-0.1, 0.8, 0.5};
    HostVector prm{0.6, -0.3, 0.9, 0.1};

    CudaContext                ctx;
    inverse::TimeObjectivePlan dev_obj;
    dev_obj.add(obj, ctx);

    state::DeviceTimeTrajectory dev_tr;
    state::copy(tr, dev_tr, ctx);
    DeviceVector dev_prm;
    copy(prm, dev_prm, ctx);
    DeviceVector dev_state_grad(3);
    DeviceVector dev_param_grad(4);

    const Real* const tr_ptr    = dev_tr.data();
    const Real* const prm_ptr   = dev_prm.data();
    const Real* const state_ptr = dev_state_grad.data();
    const Real* const param_ptr = dev_param_grad.data();

    const Real host_val  = obj.value(tr, prm);
    const Real dev_val   = dev_obj.value(dev_tr, dev_prm.view(), ctx);
    status              *= near(host_val, dev_val);

    HostVector all_state_grad(tr.size());
    for (Index level = 0; level < tr.numTimeLevels(); ++level)
    {
      HostVector host_grad;
      obj.stateGrad(level, tr, prm, host_grad);
      dev_obj.stateGrad(
          level, dev_tr, dev_prm.view(), dev_state_grad.view(), ctx);
      HostVector dev_grad;
      copy(dev_state_grad, dev_grad, ctx);
      ctx.synchronize();
      status *= near(host_grad, dev_grad);
      for (Index i = 0; i < tr.numStates(); ++i)
      {
        all_state_grad[level * tr.numStates() + i] = dev_grad[i];
      }
    }

    HostVector host_param_grad;
    obj.paramGrad(tr, prm, host_param_grad);
    dev_obj.paramGrad(
        dev_tr, dev_prm.view(), dev_param_grad.view(), ctx);
    HostVector dev_param_grad_h;
    copy(dev_param_grad, dev_param_grad_h, ctx);
    ctx.synchronize();
    status *= near(host_param_grad, dev_param_grad_h);

    const HostVector state_dir{
        0.3, -0.2, 0.1, -0.4, 0.25, 0.5, 0.15, -0.35, 0.2};
    const HostVector      param_dir{0.2, -0.1, 0.4, -0.3};
    constexpr Real        eps       = 1.0e-6;
    state::TimeTrajectory plus      = tr;
    state::TimeTrajectory minus     = tr;
    HostVector            prm_plus  = prm;
    HostVector            prm_minus = prm;
    for (Index i = 0; i < tr.size(); ++i)
    {
      plus.data()[i]  += eps * state_dir[i];
      minus.data()[i] -= eps * state_dir[i];
    }
    for (Index i = 0; i < prm.size(); ++i)
    {
      prm_plus[i]  += eps * param_dir[i];
      prm_minus[i] -= eps * param_dir[i];
    }
    const Real fd = (obj.value(plus, prm_plus)
                     - obj.value(minus, prm_minus))
                    / (2.0 * eps);
    const Real grad_dir = inner(all_state_grad, state_dir)
                          + inner(dev_param_grad_h, param_dir);
    status *= near(fd, grad_dir, 2.0e-8);

    const Real repeat = dev_obj.value(dev_tr, dev_prm.view(), ctx);
    dev_obj.paramGrad(
        dev_tr, dev_prm.view(), dev_param_grad.view(), ctx);
    ctx.synchronize();
    status *= near(repeat, host_val);
    status *= tr_ptr == dev_tr.data() && prm_ptr == dev_prm.data()
              && state_ptr == dev_state_grad.data()
              && param_ptr == dev_param_grad.data();
  }
  catch (const std::exception& err)
  {
    std::cout << "    exception: " << err.what() << '\n';
    status *= false;
  }

  return status.report();
}

TestOutcome deviceObjectiveOwnsCopiedObservation()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    CudaContext                ctx;
    inverse::TimeObjectivePlan dev_obj;
    {
      LinearObservation            host_obs;
      inverse::TimeObservationData data(1, 2);
      data[0] = HostVector{0.0, 0.0};
      data.setTimeLevels({0});
      inverse::TimeLeastSquaresObjective host_obj(host_obs, data, 1.0);
      dev_obj.add(host_obj, ctx);
    }

    state::TimeTrajectory tr(2, 3);
    tr[0] = HostVector{1.0, 2.0, 3.0};
    state::DeviceTimeTrajectory dev_tr;
    state::copy(tr, dev_tr, ctx);
    DeviceVector dev_prm(4);

    status *= near(dev_obj.value(dev_tr, dev_prm.view(), ctx), 12.625);
  }
  catch (const std::exception& err)
  {
    std::cout << "    exception: " << err.what() << '\n';
    status *= false;
  }

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results +=
      femx::tests::deviceObjectiveMatchesHostAndFiniteDifference();
  results += femx::tests::deviceObjectiveOwnsCopiedObservation();
  return results.summary();
}
