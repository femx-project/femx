#include <algorithm>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/native/HostSystemMatrix.hpp>

namespace femx
{
namespace fem
{
namespace
{
void checkInitVecs(Index                      num_states,
                   Index                      num_prm,
                   HostVectorView<const Real> in,
                   HostVectorView<Real>       out)
{
  require(in.size() == num_prm && out.size() == num_states,
          "InitialStateMap vector size mismatch");
}

void checkInitVecs(Index                        num_states,
                   Index                        num_prm,
                   DeviceVectorView<const Real> in,
                   DeviceVectorView<Real>       out)
{
  require(in.size() == num_prm && out.size() == num_states,
          "InitialStateMap Device vector size mismatch");
}

} // namespace

HostControlMap makeControlMap(
    Index                           num_steps,
    Index                           num_states,
    const DirichletControl&         ctr,
    HostVector<Index>               fixed_dofs,
    HostVector<Real>                fixed_vals,
    HostVector<LinearInterpolation> time,
    Index                           ctr_off,
    Index                           num_prm)
{
  require(num_steps > 0 && num_states > 0 && ctr_off >= 0,
          "ControlMap received invalid dimensions");

  HostVector<Index> dofs;
  dofs.reserve(ctr.numStateDofs() + fixed_dofs.size());
  HostVector<char> used(num_states, 0);
  for (Index dof : ctr.stateDofs())
  {
    require(dof >= 0 && dof < num_states && used[dof] == 0,
            "ControlMap controlled DOF is invalid");
    used[dof] = 1;
    dofs.push_back(dof);
  }
  for (Index dof : fixed_dofs)
  {
    require(dof >= 0 && dof < num_states && used[dof] == 0,
            "ControlMap fixed DOF is invalid");
    used[dof] = 1;
    dofs.push_back(dof);
  }

  if (time.empty())
  {
    time.resize(num_steps);
    for (Index step = 0; step < num_steps; ++step)
    {
      time[step] = {step, step, 0.0};
    }
  }
  require(time.size() == num_steps,
          "ControlMap time stencil count mismatch");

  HostVector<Index> lower(num_steps);
  HostVector<Index> upper(num_steps);
  HostVector<Real>  upper_wts(num_steps);
  Index             num_levels = 0;
  for (Index step = 0; step < num_steps; ++step)
  {
    require(time[step].isValid(),
            "ControlMap time stencil is invalid");

    lower[step]     = time[step].lower;
    upper[step]     = time[step].upper;
    upper_wts[step] = time[step].upper_weight;
    num_levels      = std::max(num_levels, time[step].upper + 1);
  }

  const Index required = ctr_off + num_levels * ctr.numControlParams();
  if (num_prm < 0)
  {
    num_prm = required;
  }
  require(num_prm >= required,
          "ControlMap parameter count is too small");

  const Index num_fixed = fixed_dofs.size();
  if (fixed_vals.empty())
  {
    fixed_vals.resize(num_steps * num_fixed);
  }
  else if (fixed_vals.size() == num_fixed)
  {
    linalg::HostContext ctx;
    auto&               vec_handler = ctx.vectorHandler();
    HostVector<Real>    vals(num_steps * num_fixed);

    for (Index step = 0; step < num_steps; ++step)
    {
      vec_handler.copy(fixed_vals.view(),
                       vals.view().subview(step * num_fixed, num_fixed));
    }
    fixed_vals = std::move(vals);
  }
  else
  {
    require(fixed_vals.size() == num_steps * num_fixed,
            "ControlMap fixed value size mismatch");
  }

  HostControlMap out;

  out.num_steps_  = num_steps;
  out.num_states_ = num_states;
  out.num_prm_    = num_prm;
  out.num_fixed_  = num_fixed;
  out.ctr_off_    = ctr_off;
  out.ctr_mat_    = ctr.matrix();
  out.dofs_       = std::move(dofs);
  out.fixed_vals_ = std::move(fixed_vals);
  out.lower_      = std::move(lower);
  out.upper_      = std::move(upper);
  out.upper_wts_  = std::move(upper_wts);
  out.compact_.resize(ctr.numStateDofs());

  return out;
}

void copy(const HostControlMap& src,
          DeviceControlMap&     dst,
          linalg::CudaContext&  ctx)
{
  auto& vec_handler = ctx.vectorHandler();
  dst.num_steps_    = src.num_steps_;
  dst.num_states_   = src.num_states_;
  dst.num_prm_      = src.num_prm_;
  dst.num_fixed_    = src.num_fixed_;
  dst.ctr_off_      = src.ctr_off_;

  DeviceCsrPattern pattern;
  femx::copy(src.ctr_mat_.pattern(), pattern, ctx);
  DeviceCsrMatrix mat(pattern);
  vec_handler.copy(src.ctr_mat_.vals(), mat.vals());
  dst.ctr_mat_ = std::move(mat);

  vec_handler.copy(src.dofs_, dst.dofs_);
  vec_handler.copy(src.fixed_vals_, dst.fixed_vals_);
  dst.lower_     = src.lower_;
  dst.upper_     = src.upper_;
  dst.upper_wts_ = src.upper_wts_;
  dst.compact_.resize(src.ctr_mat_.rows());
}

void controlVals(const HostControlMap&      map,
                 Index                      step,
                 HostVectorView<const Real> prm,
                 HostVectorView<Real>       out)
{
  require(step >= 0 && step < map.num_steps_ && prm.size() == map.num_prm_
              && out.size() == map.numBcs(),
          "ControlMap vector size mismatch");

  const Index lo    = map.lower_[step];
  const Index hi    = map.upper_[step];
  const Real  hi_wt = map.upper_wts_[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Index block = map.ctr_mat_.cols();

  HostVectorView<Real>     ctr_vals = out.subview(0, map.ctr_mat_.rows());
  linalg::HostContext      ctx;
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::HostSystemMatrix jac(ctx);

  jac.apply(map.ctr_mat_,
            prm.subview(map.ctr_off_ + lo * block, block),
            ctr_vals,
            lo_wt,
            0.0);

  if (hi != lo && hi_wt != 0.0)
  {
    jac.apply(map.ctr_mat_,
              prm.subview(map.ctr_off_ + hi * block, block),
              ctr_vals,
              hi_wt,

              1.0);
  }
  vec_handler.copy(map.fixed_vals_.view().subview(step * map.num_fixed_,
                                                  map.num_fixed_),
                   out.subview(map.ctr_mat_.rows(), map.num_fixed_));
}

void controlVals(const DeviceControlMap&      map,
                 Index                        step,
                 DeviceVectorView<const Real> prm,
                 DeviceVectorView<Real>       out,
                 linalg::CudaContext&         ctx)
{
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::CudaSystemMatrix jac(ctx);
  require(step >= 0 && step < map.num_steps_ && prm.size() == map.num_prm_
              && out.size() == map.numBcs(),
          "ControlMap Device vector size mismatch");

  const Index            lo       = map.lower_[step];
  const Index            hi       = map.upper_[step];
  const Real             hi_wt    = map.upper_wts_[step];
  const Real             lo_wt    = 1.0 - hi_wt;
  const Index            block    = map.ctr_mat_.cols();
  DeviceVectorView<Real> ctr_vals = out.subview(0, map.ctr_mat_.rows());
  jac.apply(map.ctr_mat_,
            prm.subview(map.ctr_off_ + lo * block, block),
            ctr_vals,
            lo_wt,
            0.0);
  if (hi != lo && hi_wt != 0.0)
  {
    jac.apply(map.ctr_mat_,
              prm.subview(map.ctr_off_ + hi * block, block),
              ctr_vals,
              hi_wt,
              1.0);
  }
  vec_handler.copy(map.fixed_vals_.view().subview(step * map.num_fixed_,
                                                  map.num_fixed_),
                   out.subview(map.ctr_mat_.rows(), map.num_fixed_));
}

void controlJac(const HostControlMap&      map,
                Index                      step,
                HostVectorView<const Real> dir,
                HostVectorView<Real>       out)
{
  require(step >= 0 && step < map.num_steps_ && dir.size() == map.num_prm_
              && out.size() == map.num_states_,
          "ControlMap Jacobian vector size mismatch");
  linalg::HostContext      ctx;
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::HostSystemMatrix jac(ctx);
  vec_handler.zero(out);

  const Index lo    = map.lower_[step];
  const Index hi    = map.upper_[step];
  const Real  hi_wt = map.upper_wts_[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Index block = map.ctr_mat_.cols();
  jac.apply(map.ctr_mat_,
            dir.subview(map.ctr_off_ + lo * block, block),
            map.compact_.view(),
            -lo_wt,
            0.0);
  if (hi != lo && hi_wt != 0.0)
  {
    jac.apply(map.ctr_mat_,
              dir.subview(map.ctr_off_ + hi * block, block),
              map.compact_.view(),
              -hi_wt,
              1.0);
  }
  vec_handler.scatter(map.compact_.view(),
                      map.dofs_.view().subview(0, map.ctr_mat_.rows()),
                      out);
}

void controlJac(const DeviceControlMap&      map,
                Index                        step,
                DeviceVectorView<const Real> dir,
                DeviceVectorView<Real>       out,
                linalg::CudaContext&         ctx)
{
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::CudaSystemMatrix jac(ctx);
  require(step >= 0 && step < map.num_steps_ && dir.size() == map.num_prm_
              && out.size() == map.num_states_,
          "ControlMap Device Jacobian size mismatch");
  vec_handler.zero(out);

  const Index lo    = map.lower_[step];
  const Index hi    = map.upper_[step];
  const Real  hi_wt = map.upper_wts_[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Index block = map.ctr_mat_.cols();
  jac.apply(map.ctr_mat_,
            dir.subview(map.ctr_off_ + lo * block, block),
            map.compact_.view(),
            -lo_wt,
            0.0);
  if (hi != lo && hi_wt != 0.0)
  {
    jac.apply(map.ctr_mat_,
              dir.subview(map.ctr_off_ + hi * block, block),
              map.compact_.view(),
              -hi_wt,
              1.0);
  }
  vec_handler.scatter(map.compact_.view(),
                      map.dofs_.view().subview(0, map.ctr_mat_.rows()),
                      out);
}

void addControlJacT(const HostControlMap&      map,
                    Index                      step,
                    HostVectorView<const Real> adj,
                    HostVectorView<Real>       grad)
{
  require(step >= 0 && step < map.num_steps_ && adj.size() == map.num_states_
              && grad.size() == map.num_prm_,
          "ControlMap transpose vector size mismatch");
  linalg::HostContext      ctx;
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::HostSystemMatrix jac(ctx);
  vec_handler.gather(adj,
                     map.dofs_.view().subview(0, map.ctr_mat_.rows()),
                     map.compact_.view());

  const Index lo    = map.lower_[step];
  const Index hi    = map.upper_[step];
  const Real  hi_wt = map.upper_wts_[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Index block = map.ctr_mat_.cols();
  jac.applyT(map.ctr_mat_,
             map.compact_.view(),
             grad.subview(map.ctr_off_ + lo * block, block),
             -lo_wt,
             1.0);
  if (hi != lo && hi_wt != 0.0)
  {
    jac.applyT(map.ctr_mat_,
               map.compact_.view(),
               grad.subview(map.ctr_off_ + hi * block, block),
               -hi_wt,
               1.0);
  }
}

void addControlJacT(const DeviceControlMap&      map,
                    Index                        step,
                    DeviceVectorView<const Real> adj,
                    DeviceVectorView<Real>       grad,
                    linalg::CudaContext&         ctx)
{
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::CudaSystemMatrix jac(ctx);
  require(step >= 0 && step < map.num_steps_ && adj.size() == map.num_states_
              && grad.size() == map.num_prm_,
          "ControlMap Device transpose size mismatch");
  vec_handler.gather(adj,
                     map.dofs_.view().subview(0, map.ctr_mat_.rows()),
                     map.compact_.view());

  const Index lo    = map.lower_[step];
  const Index hi    = map.upper_[step];
  const Real  hi_wt = map.upper_wts_[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Index block = map.ctr_mat_.cols();
  jac.applyT(map.ctr_mat_,
             map.compact_.view(),
             grad.subview(map.ctr_off_ + lo * block, block),
             -lo_wt,
             1.0);
  if (hi != lo && hi_wt != 0.0)
  {
    jac.applyT(map.ctr_mat_,
               map.compact_.view(),
               grad.subview(map.ctr_off_ + hi * block, block),
               -hi_wt,
               1.0);
  }
}

HostInitialStateMap makeInitialStateMap(HostVector<Real>        mean,
                                        DenseMatrix             modes,
                                        const DirichletControl& ctr,
                                        Index                   init_off,
                                        Index                   ctr_off,
                                        Index                   num_prm)
{
  require(!mean.empty() && modes.rows() == mean.size(),
          "InitialStateMap mean and modes must match the state size");
  require(init_off >= 0 && ctr_off >= 0 && num_prm >= 0
              && init_off + modes.cols() <= num_prm
              && ctr_off + ctr.numControlParams() <= num_prm,
          "InitialStateMap parameter blocks do not fit the parameter vector");
  for (Index row : ctr.stateDofs())
  {
    require(row >= 0 && row < mean.size(),
            "InitialStateMap controlled state DOF is out of range");
    for (Index col = 0; col < modes.cols(); ++col)
    {
      require(modes(row, col) == 0.0,
              "InitialStateMap modes must vanish on controlled DOFs");
    }
  }

  HostVector<Real>    flat_modes(modes.size());
  linalg::HostContext ctx;
  auto&               vec_handler = ctx.vectorHandler();
  vec_handler.copy(HostVectorView<const Real>(modes.data(), modes.size()),
                   flat_modes.view());
  HostInitialStateMap out;
  out.num_states_ = mean.size();
  out.num_prm_    = num_prm;
  out.num_modes_  = modes.cols();
  out.init_off_   = init_off;
  out.ctr_off_    = ctr_off;
  out.mean_       = std::move(mean);
  out.modes_      = std::move(flat_modes);
  out.ctr_mat_    = ctr.matrix();
  out.ctr_dofs_   = ctr.stateDofs();
  out.compact_.resize(ctr.numStateDofs());
  return out;
}

void copy(const HostInitialStateMap& src,
          DeviceInitialStateMap&     dst,
          linalg::CudaContext&       ctx)
{
  auto& vec_handler = ctx.vectorHandler();
  dst.num_states_   = src.num_states_;
  dst.num_prm_      = src.num_prm_;
  dst.num_modes_    = src.num_modes_;
  dst.init_off_     = src.init_off_;
  dst.ctr_off_      = src.ctr_off_;
  vec_handler.copy(src.mean_, dst.mean_);
  vec_handler.copy(src.modes_, dst.modes_);

  DeviceCsrPattern pattern;
  femx::copy(src.ctr_mat_.pattern(), pattern, ctx);
  DeviceCsrMatrix mat(pattern);
  vec_handler.copy(src.ctr_mat_.vals(), mat.vals());
  dst.ctr_mat_ = std::move(mat);

  vec_handler.copy(src.ctr_dofs_, dst.ctr_dofs_);
  dst.compact_.resize(src.ctr_mat_.rows());
}

void initialState(const HostInitialStateMap& map,
                  HostVectorView<const Real> prm,
                  HostVectorView<Real>       out)
{
  checkInitVecs(map.num_states_, map.num_prm_, prm, out);
  linalg::HostContext      ctx;
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::HostSystemMatrix jac(ctx);
  vec_handler.copy(map.mean_.view(), out);
  if (map.num_modes_ > 0)
  {
    jac.apply(HostMatrixView<const Real>(map.modes_.data(),
                                         map.num_states_,
                                         map.num_modes_),
              prm.subview(map.init_off_, map.num_modes_),
              out,
              1.0,
              1.0);
  }
  if (map.ctr_mat_.rows() > 0)
  {
    jac.apply(map.ctr_mat_,
              prm.subview(map.ctr_off_, map.ctr_mat_.cols()),
              map.compact_.view());
    vec_handler.scatter(map.compact_.view(), map.ctr_dofs_.view(), out);
  }
}

void initialState(const DeviceInitialStateMap& map,
                  DeviceVectorView<const Real> prm,
                  DeviceVectorView<Real>       out,
                  linalg::CudaContext&         ctx)
{
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::CudaSystemMatrix jac(ctx);
  checkInitVecs(map.num_states_, map.num_prm_, prm, out);
  vec_handler.copy(map.mean_.view(), out);
  if (map.num_modes_ > 0)
  {
    jac.apply(DeviceMatrixView<const Real>(map.modes_.data(),
                                           map.num_states_,
                                           map.num_modes_),
              prm.subview(map.init_off_, map.num_modes_),
              out,
              1.0,
              1.0);
  }
  if (map.ctr_mat_.rows() > 0)
  {
    jac.apply(map.ctr_mat_,
              prm.subview(map.ctr_off_, map.ctr_mat_.cols()),
              map.compact_.view());
    vec_handler.scatter(map.compact_.view(), map.ctr_dofs_.view(), out);
  }
}

void addInitialJacT(const HostInitialStateMap& map,
                    HostVectorView<const Real> adj,
                    HostVectorView<Real>       grad)
{
  checkInitVecs(map.num_prm_, map.num_states_, adj, grad);
  linalg::HostContext      ctx;
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::HostSystemMatrix jac(ctx);
  if (map.num_modes_ > 0)
  {
    jac.applyT(HostMatrixView<const Real>(map.modes_.data(),
                                          map.num_states_,
                                          map.num_modes_),
               adj,
               grad.subview(map.init_off_, map.num_modes_),
               1.0,
               1.0);
  }
  if (map.ctr_mat_.rows() > 0)
  {
    vec_handler.gather(adj, map.ctr_dofs_.view(), map.compact_.view());
    jac.applyT(map.ctr_mat_,
               map.compact_.view(),
               grad.subview(map.ctr_off_, map.ctr_mat_.cols()),
               1.0,
               1.0);
  }
}

void addInitialJacT(const DeviceInitialStateMap& map,
                    DeviceVectorView<const Real> adj,
                    DeviceVectorView<Real>       grad,
                    linalg::CudaContext&         ctx)
{
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::CudaSystemMatrix jac(ctx);
  checkInitVecs(map.num_prm_, map.num_states_, adj, grad);
  if (map.num_modes_ > 0)
  {
    jac.applyT(DeviceMatrixView<const Real>(map.modes_.data(),
                                            map.num_states_,
                                            map.num_modes_),
               adj,
               grad.subview(map.init_off_, map.num_modes_),
               1.0,
               1.0);
  }
  if (map.ctr_mat_.rows() > 0)
  {
    vec_handler.gather(adj, map.ctr_dofs_.view(), map.compact_.view());
    jac.applyT(map.ctr_mat_,
               map.compact_.view(),
               grad.subview(map.ctr_off_, map.ctr_mat_.cols()),
               1.0,
               1.0);
  }
}

} // namespace fem
} // namespace femx
