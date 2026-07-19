#include <algorithm>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/fem/ControlMap.hpp>

namespace femx
{
namespace fem
{
namespace
{
void checkInitVecs(Index               num_states,
                   Index               num_prm,
                   HostConstVectorView in,
                   HostVectorView      out)
{
  require(in.size() == num_prm && out.size() == num_states,
          "InitialStateMap vector size mismatch");
}

void checkInitVecs(Index                 num_states,
                   Index                 num_prm,
                   DeviceConstVectorView in,
                   DeviceVectorView      out)
{
  require(in.size() == num_prm && out.size() == num_states,
          "InitialStateMap Device vector size mismatch");
}

} // namespace

HostControlMap makeControlMap(
    Index                      num_steps,
    Index                      num_states,
    const DirichletControl&    ctr,
    Array<Index>               fixed_dofs,
    HostVector                 fixed_vals,
    Array<LinearInterpolation> time,
    Index                      ctr_off,
    Index                      num_prm)
{
  require(num_steps > 0 && num_states > 0 && ctr_off >= 0,
          "ControlMap received invalid dimensions");

  HostIndexVector dofs;
  dofs.reserve(ctr.numStateDofs() + fixed_dofs.size());
  Array<char> used(num_states, 0);
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

  HostIndexVector lower(num_steps);
  HostIndexVector upper(num_steps);
  HostVector      upper_wts(num_steps);
  Index           num_levels = 0;
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
    CpuContext ctx;
    HostVector vals(num_steps * num_fixed);
    for (Index step = 0; step < num_steps; ++step)
    {
      copy(fixed_vals.view(),
           vals.view().subview(step * num_fixed, num_fixed),
           ctx);
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
  out.control_    = ctr.matrix();
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
          CudaContext&          ctx)
{
  dst.num_steps_  = src.num_steps_;
  dst.num_states_ = src.num_states_;
  dst.num_prm_    = src.num_prm_;
  dst.num_fixed_  = src.num_fixed_;
  dst.ctr_off_    = src.ctr_off_;

  DeviceCsrGraph graph;
  femx::copy(src.control_.graph(), graph, ctx);
  DeviceCsrMatrix mat(graph);
  femx::copy(src.control_, mat, ctx);
  dst.control_ = std::move(mat);

  femx::copy(src.dofs_, dst.dofs_, ctx);
  femx::copy(src.fixed_vals_, dst.fixed_vals_, ctx);
  dst.lower_     = src.lower_;
  dst.upper_     = src.upper_;
  dst.upper_wts_ = src.upper_wts_;
  dst.compact_.resize(src.control_.rows());
}

void controlVals(const HostControlMap& map,
                 Index                 step,
                 HostConstVectorView   prm,
                 HostVectorView        out)
{
  require(step >= 0 && step < map.num_steps_ && prm.size() == map.num_prm_
              && out.size() == map.numBcs(),
          "ControlMap vector size mismatch");

  const Index    lo         = map.lower_[step];
  const Index    hi         = map.upper_[step];
  const Real     hi_wt      = map.upper_wts_[step];
  const Real     lo_wt      = 1.0 - hi_wt;
  const Index    block      = map.control_.cols();
  HostVectorView controlled = out.subview(0, map.control_.rows());
  CpuContext     ctx;
  apply(map.control_,
        prm.subview(map.ctr_off_ + lo * block, block),
        controlled,
        ctx,
        lo_wt,
        0.0);
  if (hi != lo && hi_wt != 0.0)
  {
    apply(map.control_,
          prm.subview(map.ctr_off_ + hi * block, block),
          controlled,
          ctx,
          hi_wt,
          1.0);
  }
  copy(map.fixed_vals_.view().subview(step * map.num_fixed_,
                                      map.num_fixed_),
       out.subview(map.control_.rows(), map.num_fixed_),
       ctx);
}

void controlVals(const DeviceControlMap& map,
                 Index                   step,
                 DeviceConstVectorView   prm,
                 DeviceVectorView        out,
                 CudaContext&            ctx)
{
  require(step >= 0 && step < map.num_steps_ && prm.size() == map.num_prm_
              && out.size() == map.numBcs(),
          "ControlMap Device vector size mismatch");

  const Index      lo         = map.lower_[step];
  const Index      hi         = map.upper_[step];
  const Real       hi_wt      = map.upper_wts_[step];
  const Real       lo_wt      = 1.0 - hi_wt;
  const Index      block      = map.control_.cols();
  DeviceVectorView controlled = out.subview(0, map.control_.rows());
  apply(map.control_,
        prm.subview(map.ctr_off_ + lo * block, block),
        controlled,
        ctx,
        lo_wt,
        0.0);
  if (hi != lo && hi_wt != 0.0)
  {
    apply(map.control_,
          prm.subview(map.ctr_off_ + hi * block, block),
          controlled,
          ctx,
          hi_wt,
          1.0);
  }
  femx::copy(map.fixed_vals_.view().subview(step * map.num_fixed_,
                                            map.num_fixed_),
             out.subview(map.control_.rows(), map.num_fixed_),
             ctx);
}

void controlJac(const HostControlMap& map,
                Index                 step,
                HostConstVectorView   dir,
                HostVectorView        out)
{
  require(step >= 0 && step < map.num_steps_ && dir.size() == map.num_prm_
              && out.size() == map.num_states_,
          "ControlMap Jacobian vector size mismatch");
  CpuContext ctx;
  zero(out, ctx);

  const Index lo    = map.lower_[step];
  const Index hi    = map.upper_[step];
  const Real  hi_wt = map.upper_wts_[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Index block = map.control_.cols();
  apply(map.control_,
        dir.subview(map.ctr_off_ + lo * block, block),
        map.compact_.view(),
        ctx,
        -lo_wt,
        0.0);
  if (hi != lo && hi_wt != 0.0)
  {
    apply(map.control_,
          dir.subview(map.ctr_off_ + hi * block, block),
          map.compact_.view(),
          ctx,
          -hi_wt,
          1.0);
  }
  scatter(map.compact_.view(),
          map.dofs_.view().subview(0, map.control_.rows()),
          out,
          ctx);
}

void controlJac(const DeviceControlMap& map,
                Index                   step,
                DeviceConstVectorView   dir,
                DeviceVectorView        out,
                CudaContext&            ctx)
{
  require(step >= 0 && step < map.num_steps_ && dir.size() == map.num_prm_
              && out.size() == map.num_states_,
          "ControlMap Device Jacobian size mismatch");
  zero(out, ctx);

  const Index lo    = map.lower_[step];
  const Index hi    = map.upper_[step];
  const Real  hi_wt = map.upper_wts_[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Index block = map.control_.cols();
  apply(map.control_,
        dir.subview(map.ctr_off_ + lo * block, block),
        map.compact_.view(),
        ctx,
        -lo_wt,
        0.0);
  if (hi != lo && hi_wt != 0.0)
  {
    apply(map.control_,
          dir.subview(map.ctr_off_ + hi * block, block),
          map.compact_.view(),
          ctx,
          -hi_wt,
          1.0);
  }
  scatter(map.compact_.view(),
          map.dofs_.view().subview(0, map.control_.rows()),
          out,
          ctx);
}

void addControlJacT(const HostControlMap& map,
                    Index                 step,
                    HostConstVectorView   adj,
                    HostVectorView        grad)
{
  require(step >= 0 && step < map.num_steps_ && adj.size() == map.num_states_
              && grad.size() == map.num_prm_,
          "ControlMap transpose vector size mismatch");
  CpuContext ctx;
  gather(adj,
         map.dofs_.view().subview(0, map.control_.rows()),
         map.compact_.view(),
         ctx);

  const Index lo    = map.lower_[step];
  const Index hi    = map.upper_[step];
  const Real  hi_wt = map.upper_wts_[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Index block = map.control_.cols();
  applyT(map.control_,
         map.compact_.view(),
         grad.subview(map.ctr_off_ + lo * block, block),
         ctx,
         -lo_wt,
         1.0);
  if (hi != lo && hi_wt != 0.0)
  {
    applyT(map.control_,
           map.compact_.view(),
           grad.subview(map.ctr_off_ + hi * block, block),
           ctx,
           -hi_wt,
           1.0);
  }
}

void addControlJacT(const DeviceControlMap& map,
                    Index                   step,
                    DeviceConstVectorView   adj,
                    DeviceVectorView        grad,
                    CudaContext&            ctx)
{
  require(step >= 0 && step < map.num_steps_ && adj.size() == map.num_states_
              && grad.size() == map.num_prm_,
          "ControlMap Device transpose size mismatch");
  gather(adj,
         map.dofs_.view().subview(0, map.control_.rows()),
         map.compact_.view(),
         ctx);

  const Index lo    = map.lower_[step];
  const Index hi    = map.upper_[step];
  const Real  hi_wt = map.upper_wts_[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Index block = map.control_.cols();
  applyT(map.control_,
         map.compact_.view(),
         grad.subview(map.ctr_off_ + lo * block, block),
         ctx,
         -lo_wt,
         1.0);
  if (hi != lo && hi_wt != 0.0)
  {
    applyT(map.control_,
           map.compact_.view(),
           grad.subview(map.ctr_off_ + hi * block, block),
           ctx,
           -hi_wt,
           1.0);
  }
}

HostInitialStateMap makeInitialStateMap(HostVector              mean,
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

  HostVector flat_modes(modes.size());
  CpuContext ctx;
  copy(HostConstVectorView(modes.data(), modes.size()),
       flat_modes.view(),
       ctx);
  HostInitialStateMap out;
  out.num_states_ = mean.size();
  out.num_prm_    = num_prm;
  out.num_modes_  = modes.cols();
  out.init_off_   = init_off;
  out.ctr_off_    = ctr_off;
  out.mean_       = std::move(mean);
  out.modes_      = std::move(flat_modes);
  out.control_    = ctr.matrix();
  out.ctr_dofs_   = ctr.stateDofs();
  out.compact_.resize(ctr.numStateDofs());
  return out;
}

void copy(const HostInitialStateMap& src,
          DeviceInitialStateMap&     dst,
          CudaContext&               ctx)
{
  dst.num_states_ = src.num_states_;
  dst.num_prm_    = src.num_prm_;
  dst.num_modes_  = src.num_modes_;
  dst.init_off_   = src.init_off_;
  dst.ctr_off_    = src.ctr_off_;
  femx::copy(src.mean_, dst.mean_, ctx);
  femx::copy(src.modes_, dst.modes_, ctx);

  DeviceCsrGraph graph;
  femx::copy(src.control_.graph(), graph, ctx);
  DeviceCsrMatrix mat(graph);
  femx::copy(src.control_, mat, ctx);
  dst.control_ = std::move(mat);

  femx::copy(src.ctr_dofs_, dst.ctr_dofs_, ctx);
  dst.compact_.resize(src.control_.rows());
}

void initialState(const HostInitialStateMap& map,
                  HostConstVectorView        prm,
                  HostVectorView             out)
{
  checkInitVecs(map.num_states_, map.num_prm_, prm, out);
  CpuContext ctx;
  copy(map.mean_.view(), out, ctx);
  if (map.num_modes_ > 0)
  {
    apply(HostMatrixView<const Real>(map.modes_.data(),
                                     map.num_states_,
                                     map.num_modes_),
          prm.subview(map.init_off_, map.num_modes_),
          out,
          ctx,
          1.0,
          1.0);
  }
  if (map.control_.rows() > 0)
  {
    apply(map.control_,
          prm.subview(map.ctr_off_, map.control_.cols()),
          map.compact_.view(),
          ctx);
    scatter(map.compact_.view(), map.ctr_dofs_.view(), out, ctx);
  }
}

void initialState(const DeviceInitialStateMap& map,
                  DeviceConstVectorView        prm,
                  DeviceVectorView             out,
                  CudaContext&                 ctx)
{
  checkInitVecs(map.num_states_, map.num_prm_, prm, out);
  femx::copy(map.mean_.view(), out, ctx);
  if (map.num_modes_ > 0)
  {
    apply(DeviceMatrixView<const Real>(map.modes_.data(),
                                       map.num_states_,
                                       map.num_modes_),
          prm.subview(map.init_off_, map.num_modes_),
          out,
          ctx,
          1.0,
          1.0);
  }
  if (map.control_.rows() > 0)
  {
    apply(map.control_,
          prm.subview(map.ctr_off_, map.control_.cols()),
          map.compact_.view(),
          ctx);
    scatter(map.compact_.view(), map.ctr_dofs_.view(), out, ctx);
  }
}

void addInitialJacT(const HostInitialStateMap& map,
                    HostConstVectorView        adj,
                    HostVectorView             grad)
{
  checkInitVecs(map.num_prm_, map.num_states_, adj, grad);
  CpuContext ctx;
  if (map.num_modes_ > 0)
  {
    applyT(HostMatrixView<const Real>(map.modes_.data(),
                                      map.num_states_,
                                      map.num_modes_),
           adj,
           grad.subview(map.init_off_, map.num_modes_),
           ctx,
           1.0,
           1.0);
  }
  if (map.control_.rows() > 0)
  {
    gather(adj, map.ctr_dofs_.view(), map.compact_.view(), ctx);
    applyT(map.control_,
           map.compact_.view(),
           grad.subview(map.ctr_off_, map.control_.cols()),
           ctx,
           1.0,
           1.0);
  }
}

void addInitialJacT(const DeviceInitialStateMap& map,
                    DeviceConstVectorView        adj,
                    DeviceVectorView             grad,
                    CudaContext&                 ctx)
{
  checkInitVecs(map.num_prm_, map.num_states_, adj, grad);
  if (map.num_modes_ > 0)
  {
    applyT(DeviceMatrixView<const Real>(map.modes_.data(),
                                        map.num_states_,
                                        map.num_modes_),
           adj,
           grad.subview(map.init_off_, map.num_modes_),
           ctx,
           1.0,
           1.0);
  }
  if (map.control_.rows() > 0)
  {
    gather(adj, map.ctr_dofs_.view(), map.compact_.view(), ctx);
    applyT(map.control_,
           map.compact_.view(),
           grad.subview(map.ctr_off_, map.control_.cols()),
           ctx,
           1.0,
           1.0);
  }
}

} // namespace fem
} // namespace femx
