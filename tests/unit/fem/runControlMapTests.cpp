#include <cmath>

#include "TestHelper.hpp"
#include <femx/fem/ControlMap.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx
{
namespace tests
{
namespace
{

bool near(Real lhs, Real rhs, Real tol = 1.0e-11)
{
  return std::abs(lhs - rhs) <= tol;
}

bool near(const HostVector& lhs,
          const HostVector& rhs,
          Real              tol = 1.0e-11)
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

bool same(const HostIndexVector& lhs, const HostIndexVector& rhs)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }
  for (Index i = 0; i < lhs.size(); ++i)
  {
    if (lhs[i] != rhs[i])
    {
      return false;
    }
  }
  return true;
}

Real dot(const HostVector& lhs, const HostVector& rhs)
{
  Real val = 0.0;
  for (Index i = 0; i < lhs.size(); ++i)
  {
    val += lhs[i] * rhs[i];
  }
  return val;
}

fem::DirichletControl makeControl()
{
  return fem::DirichletControl(
      Array<Index>{1, 4},
      2,
      Array<fem::DirichletControlMapEntry>{
          {0, 0, 2.0}, {0, 1, -1.0}, {1, 0, 0.5}, {1, 1, 3.0}});
}

fem::HostControlMap makeTimeMap()
{
  return fem::makeControlMap(
      3,
      5,
      makeControl(),
      Array<Index>{0, 3},
      HostVector{10.0, 11.0, 12.0, 13.0, 14.0, 15.0},
      Array<LinearInterpolation>{
          {0, 0, 0.0}, {0, 1, 0.25}, {1, 2, 0.5}},
      2,
      9);
}

fem::HostInitialStateMap makeInitialMap()
{
  DenseMatrix modes(5, 2);
  modes(0, 0) = 1.0;
  modes(0, 1) = -0.5;
  modes(2, 0) = 2.0;
  modes(2, 1) = 1.5;
  modes(3, 0) = -1.0;
  modes(3, 1) = 0.25;
  return fem::makeInitialStateMap(
      HostVector{5.0, 6.0, 7.0, 8.0, 9.0},
      std::move(modes),
      makeControl(),
      0,
      2,
      9);
}

HostVector boundaryRes(const fem::HostControlMap& map,
                       const HostVector&          state,
                       const HostVector&          vals)
{
  HostVector res(map.numStates());
  for (Index ib = 0; ib < map.numBcs(); ++ib)
  {
    res[map.dofs()[ib]] = state[map.dofs()[ib]] - vals[ib];
  }
  return res;
}

TestOutcome hostControlMapJacobian()
{
  TestStatus                status(__func__);
  const fem::HostControlMap map = makeTimeMap();
  const HostVector          prm{0.75, -1.25, 1.0, 2.0, 3.0, -2.0, 5.0, 4.0, 8.0};
  const HostVector          dir{-0.5, 0.25, 1.0, -2.0, 0.75, 1.5, -1.0, 0.5, 3.0};
  const HostVector          adj{0.4, -1.25, 2.0, 0.75, -0.5};

  HostVector vals(map.numBcs());
  fem::controlVals(map, 1, prm.view(), vals.view());
  status *= near(vals, HostVector{2.0, 3.75, 12.0, 13.0});
  status *= same(map.dofs(), HostIndexVector{1, 4, 0, 3});

  HostVector jac(map.numStates());
  fem::controlJac(map, 1, dir.view(), jac.view());

  constexpr Real eps   = 1.0e-6;
  HostVector     plus  = prm;
  HostVector     minus = prm;
  for (Index i = 0; i < prm.size(); ++i)
  {
    plus[i]  += eps * dir[i];
    minus[i] -= eps * dir[i];
  }
  HostVector plus_vals(map.numBcs());
  HostVector minus_vals(map.numBcs());
  fem::controlVals(map, 1, plus.view(), plus_vals.view());
  fem::controlVals(map, 1, minus.view(), minus_vals.view());
  const HostVector state{2.0, 3.0, 4.0, 5.0, 6.0};
  const HostVector plus_res  = boundaryRes(map, state, plus_vals);
  const HostVector minus_res = boundaryRes(map, state, minus_vals);
  HostVector       fd(map.numStates());
  for (Index i = 0; i < fd.size(); ++i)
  {
    fd[i] = (plus_res[i] - minus_res[i]) / (2.0 * eps);
  }

  HostVector grad(map.numParams());
  fem::addControlJacT(map, 1, adj.view(), grad.view());
  status *= near(jac, fd, 2.0e-9);
  status *= near(dot(jac, adj), dot(dir, grad));
  return status.report();
}

TestOutcome hostInitialStateTranspose()
{
  TestStatus                     status(__func__);
  const fem::HostInitialStateMap map = makeInitialMap();
  const HostVector               prm{0.75, -1.25, 1.0, 2.0, 3.0, -2.0, 5.0, 4.0, 8.0};
  const HostVector               dir{-0.5, 0.25, 1.0, -2.0, 0.75, 1.5, -1.0, 0.5, 3.0};
  const HostVector               adj{0.4, -1.25, 2.0, 0.75, -0.5};

  HostVector state(map.numStates());
  fem::initialState(map, prm.view(), state.view());
  status *= near(state, HostVector{6.375, 0.0, 6.625, 6.9375, 6.5});

  constexpr Real eps   = 1.0e-6;
  HostVector     plus  = prm;
  HostVector     minus = prm;
  for (Index i = 0; i < prm.size(); ++i)
  {
    plus[i]  += eps * dir[i];
    minus[i] -= eps * dir[i];
  }
  HostVector plus_state(map.numStates());
  HostVector minus_state(map.numStates());
  fem::initialState(map, plus.view(), plus_state.view());
  fem::initialState(map, minus.view(), minus_state.view());
  HostVector jac(map.numStates());
  for (Index i = 0; i < jac.size(); ++i)
  {
    jac[i] = (plus_state[i] - minus_state[i]) / (2.0 * eps);
  }

  HostVector grad(map.numParams());
  fem::addInitialJacT(map, adj.view(), grad.view());
  status *= near(dot(jac, adj), dot(dir, grad), 2.0e-9);
  return status.report();
}

#if defined(FEMX_HAS_CUDA)
TestOutcome cudaMapsMatchHost()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  const fem::HostControlMap      host_ctr  = makeTimeMap();
  const fem::HostInitialStateMap host_init = makeInitialMap();
  const HostVector               prm{0.75, -1.25, 1.0, 2.0, 3.0, -2.0, 5.0, 4.0, 8.0};
  const HostVector               dir{-0.5, 0.25, 1.0, -2.0, 0.75, 1.5, -1.0, 0.5, 3.0};
  const HostVector               adj{0.4, -1.25, 2.0, 0.75, -0.5};

  HostVector expected_vals(host_ctr.numBcs());
  HostVector expected_jac(host_ctr.numStates());
  HostVector expected_ctr_grad(host_ctr.numParams());
  HostVector expected_state(host_init.numStates());
  HostVector expected_init_grad(host_init.numParams());
  fem::controlVals(host_ctr, 2, prm.view(), expected_vals.view());
  fem::controlJac(host_ctr, 2, dir.view(), expected_jac.view());
  fem::addControlJacT(
      host_ctr, 2, adj.view(), expected_ctr_grad.view());
  fem::initialState(host_init, prm.view(), expected_state.view());
  fem::addInitialJacT(
      host_init, adj.view(), expected_init_grad.view());

  CudaContext                ctx;
  linalg::CudaVectorHandler  vec_handler(ctx);
  fem::DeviceControlMap      ctr;
  fem::DeviceInitialStateMap init;
  DeviceVector               dev_prm;
  DeviceVector               dev_dir;
  DeviceVector               dev_adj;
  DeviceVector               vals(host_ctr.numBcs());
  DeviceVector               jac(host_ctr.numStates());
  DeviceVector               ctr_grad(host_ctr.numParams());
  DeviceVector               state(host_init.numStates());
  DeviceVector               init_grad(host_init.numParams());
  fem::copy(host_ctr, ctr, ctx);
  fem::copy(host_init, init, ctx);
  vec_handler.copy(prm, dev_prm);
  vec_handler.copy(dir, dev_dir);
  vec_handler.copy(adj, dev_adj);
  vec_handler.zero(ctr_grad.view());
  vec_handler.zero(init_grad.view());

  const Real* vals_ptr      = vals.data();
  const Real* jac_ptr       = jac.data();
  const Real* ctr_grad_ptr  = ctr_grad.data();
  const Real* state_ptr     = state.data();
  const Real* init_grad_ptr = init_grad.data();
  fem::controlVals(ctr, 2, dev_prm.view(), vals.view(), ctx);
  fem::controlJac(ctr, 2, dev_dir.view(), jac.view(), ctx);
  fem::addControlJacT(
      ctr, 2, dev_adj.view(), ctr_grad.view(), ctx);
  fem::initialState(init, dev_prm.view(), state.view(), ctx);
  fem::addInitialJacT(
      init, dev_adj.view(), init_grad.view(), ctx);

  HostVector got_vals;
  HostVector got_jac;
  HostVector got_ctr_grad;
  HostVector got_state;
  HostVector got_init_grad;
  vec_handler.copy(vals, got_vals);
  vec_handler.copy(jac, got_jac);
  vec_handler.copy(ctr_grad, got_ctr_grad);
  vec_handler.copy(state, got_state);
  vec_handler.copy(init_grad, got_init_grad);

  fem::controlVals(ctr, 2, dev_prm.view(), vals.view(), ctx);
  fem::controlJac(ctr, 2, dev_dir.view(), jac.view(), ctx);
  fem::initialState(init, dev_prm.view(), state.view(), ctx);
  ctx.sync();

  status *= near(got_vals, expected_vals);
  status *= near(got_jac, expected_jac);
  status *= near(got_ctr_grad, expected_ctr_grad);
  status *= near(dot(got_jac, adj), dot(dir, got_ctr_grad));
  status *= near(got_state, expected_state);
  status *= near(got_init_grad, expected_init_grad);
  status *= vals.data() == vals_ptr && jac.data() == jac_ptr
            && ctr_grad.data() == ctr_grad_ptr && state.data() == state_ptr
            && init_grad.data() == init_grad_ptr;
  return status.report();
}
#endif

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::hostControlMapJacobian();
  results += femx::tests::hostInitialStateTranspose();
#if defined(FEMX_HAS_CUDA)
  results += femx::tests::cudaMapsMatchHost();
#endif
  return results.summary();
}
