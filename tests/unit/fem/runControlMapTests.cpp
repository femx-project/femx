#include <cmath>

#include "TestHelper.hpp"
#include <femx/fem/ControlMap.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>

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

bool near(const HostVector<Real>& lhs,
          const HostVector<Real>& rhs,
          Real                    tol = 1.0e-11)
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

bool same(const HostVector<Index>& lhs, const HostVector<Index>& rhs)
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

Real dot(const HostVector<Real>& lhs, const HostVector<Real>& rhs)
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
      HostVector<Index>{1, 4},
      2,
      HostVector<fem::DirichletControlMapEntry>{
          {0, 0, 2.0}, {0, 1, -1.0}, {1, 0, 0.5}, {1, 1, 3.0}});
}

fem::HostControlMap makeTimeMap()
{
  return fem::makeControlMap(
      3,
      5,
      makeControl(),
      HostVector<Index>{0, 3},
      HostVector<Real>{10.0, 11.0, 12.0, 13.0, 14.0, 15.0},
      HostVector<LinearInterpolation>{
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
      HostVector<Real>{5.0, 6.0, 7.0, 8.0, 9.0},
      std::move(modes),
      makeControl(),
      0,
      2,
      9);
}

HostVector<Real> boundaryRes(const fem::HostControlMap& map,
                             const HostVector<Real>&    state,
                             const HostVector<Real>&    vals)
{
  HostVector<Real> res(map.numStates());
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
  const HostVector<Real>    prm{0.75, -1.25, 1.0, 2.0, 3.0, -2.0, 5.0, 4.0, 8.0};
  const HostVector<Real>    dir{-0.5, 0.25, 1.0, -2.0, 0.75, 1.5, -1.0, 0.5, 3.0};
  const HostVector<Real>    adj{0.4, -1.25, 2.0, 0.75, -0.5};

  HostVector<Real> vals(map.numBcs());
  fem::controlVals(map, 1, prm.view(), vals.view());
  status *= near(vals, HostVector<Real>{2.0, 3.75, 12.0, 13.0});
  status *= same(map.dofs(), HostVector<Index>{1, 4, 0, 3});

  HostVector<Real> jac(map.numStates());
  fem::controlJac(map, 1, dir.view(), jac.view());

  constexpr Real   eps   = 1.0e-6;
  HostVector<Real> plus  = prm;
  HostVector<Real> minus = prm;
  for (Index i = 0; i < prm.size(); ++i)
  {
    plus[i]  += eps * dir[i];
    minus[i] -= eps * dir[i];
  }
  HostVector<Real> plus_vals(map.numBcs());
  HostVector<Real> minus_vals(map.numBcs());
  fem::controlVals(map, 1, plus.view(), plus_vals.view());
  fem::controlVals(map, 1, minus.view(), minus_vals.view());
  const HostVector<Real> state{2.0, 3.0, 4.0, 5.0, 6.0};
  const HostVector<Real> plus_res  = boundaryRes(map, state, plus_vals);
  const HostVector<Real> minus_res = boundaryRes(map, state, minus_vals);
  HostVector<Real>       fd(map.numStates());
  for (Index i = 0; i < fd.size(); ++i)
  {
    fd[i] = (plus_res[i] - minus_res[i]) / (2.0 * eps);
  }

  HostVector<Real> grad(map.numParams());
  fem::addControlJacT(map, 1, adj.view(), grad.view());
  status *= near(jac, fd, 2.0e-9);
  status *= near(dot(jac, adj), dot(dir, grad));
  return status.report();
}

TestOutcome hostInitialStateTranspose()
{
  TestStatus                     status(__func__);
  const fem::HostInitialStateMap map = makeInitialMap();
  const HostVector<Real>         prm{0.75, -1.25, 1.0, 2.0, 3.0, -2.0, 5.0, 4.0, 8.0};
  const HostVector<Real>         dir{-0.5, 0.25, 1.0, -2.0, 0.75, 1.5, -1.0, 0.5, 3.0};
  const HostVector<Real>         adj{0.4, -1.25, 2.0, 0.75, -0.5};

  HostVector<Real> state(map.numStates());
  fem::initialState(map, prm.view(), state.view());
  status *= near(state, HostVector<Real>{6.375, 0.0, 6.625, 6.9375, 6.5});

  constexpr Real   eps   = 1.0e-6;
  HostVector<Real> plus  = prm;
  HostVector<Real> minus = prm;
  for (Index i = 0; i < prm.size(); ++i)
  {
    plus[i]  += eps * dir[i];
    minus[i] -= eps * dir[i];
  }
  HostVector<Real> plus_state(map.numStates());
  HostVector<Real> minus_state(map.numStates());
  fem::initialState(map, plus.view(), plus_state.view());
  fem::initialState(map, minus.view(), minus_state.view());
  HostVector<Real> jac(map.numStates());
  for (Index i = 0; i < jac.size(); ++i)
  {
    jac[i] = (plus_state[i] - minus_state[i]) / (2.0 * eps);
  }

  HostVector<Real> grad(map.numParams());
  fem::addInitialJacT(map, adj.view(), grad.view());
  status *= near(dot(jac, adj), dot(dir, grad), 2.0e-9);
  return status.report();
}

#if defined(FEMX_HAS_CUDA)
TestOutcome cudaMapsMatchHost()
{
  TestStatus status(__func__);
  if (!linalg::CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  const fem::HostControlMap      h_ctr  = makeTimeMap();
  const fem::HostInitialStateMap h_init = makeInitialMap();
  const HostVector<Real>         prm{0.75, -1.25, 1.0, 2.0, 3.0, -2.0, 5.0, 4.0, 8.0};
  const HostVector<Real>         dir{-0.5, 0.25, 1.0, -2.0, 0.75, 1.5, -1.0, 0.5, 3.0};
  const HostVector<Real>         adj{0.4, -1.25, 2.0, 0.75, -0.5};

  HostVector<Real> expected_vals(h_ctr.numBcs());
  HostVector<Real> expected_jac(h_ctr.numStates());
  HostVector<Real> expected_ctr_grad(h_ctr.numParams());
  HostVector<Real> expected_state(h_init.numStates());
  HostVector<Real> expected_init_grad(h_init.numParams());
  fem::controlVals(h_ctr, 2, prm.view(), expected_vals.view());
  fem::controlJac(h_ctr, 2, dir.view(), expected_jac.view());
  fem::addControlJacT(
      h_ctr, 2, adj.view(), expected_ctr_grad.view());
  fem::initialState(h_init, prm.view(), expected_state.view());
  fem::addInitialJacT(
      h_init, adj.view(), expected_init_grad.view());

  linalg::CudaContext        ctx;
  auto&                      vec_handler = ctx.vectorHandler();
  fem::DeviceControlMap      ctr;
  fem::DeviceInitialStateMap init;
  DeviceVector<Real>         d_prm;
  DeviceVector<Real>         d_dir;
  DeviceVector<Real>         d_adj;
  DeviceVector<Real>         vals(h_ctr.numBcs());
  DeviceVector<Real>         jac(h_ctr.numStates());
  DeviceVector<Real>         ctr_grad(h_ctr.numParams());
  DeviceVector<Real>         state(h_init.numStates());
  DeviceVector<Real>         init_grad(h_init.numParams());
  fem::copy(h_ctr, ctr, ctx);
  fem::copy(h_init, init, ctx);
  vec_handler.copy(prm, d_prm);
  vec_handler.copy(dir, d_dir);
  vec_handler.copy(adj, d_adj);
  vec_handler.zero(ctr_grad.view());
  vec_handler.zero(init_grad.view());

  const Real* vals_ptr      = vals.data();
  const Real* jac_ptr       = jac.data();
  const Real* ctr_grad_ptr  = ctr_grad.data();
  const Real* state_ptr     = state.data();
  const Real* init_grad_ptr = init_grad.data();
  fem::controlVals(ctr, 2, d_prm.view(), vals.view(), ctx);
  fem::controlJac(ctr, 2, d_dir.view(), jac.view(), ctx);
  fem::addControlJacT(
      ctr, 2, d_adj.view(), ctr_grad.view(), ctx);
  fem::initialState(init, d_prm.view(), state.view(), ctx);
  fem::addInitialJacT(
      init, d_adj.view(), init_grad.view(), ctx);

  HostVector<Real> got_vals;
  HostVector<Real> got_jac;
  HostVector<Real> got_ctr_grad;
  HostVector<Real> got_state;
  HostVector<Real> got_init_grad;
  vec_handler.copy(vals, got_vals);
  vec_handler.copy(jac, got_jac);
  vec_handler.copy(ctr_grad, got_ctr_grad);
  vec_handler.copy(state, got_state);
  vec_handler.copy(init_grad, got_init_grad);

  fem::controlVals(ctr, 2, d_prm.view(), vals.view(), ctx);
  fem::controlJac(ctr, 2, d_dir.view(), jac.view(), ctx);
  fem::initialState(init, d_prm.view(), state.view(), ctx);
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
