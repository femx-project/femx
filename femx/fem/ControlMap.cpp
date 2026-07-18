#include <algorithm>
#include <stdexcept>
#include <utility>

#include <femx/fem/ControlMap.hpp>

namespace femx
{
namespace fem
{
namespace
{

struct FlatControl
{
  HostIndexVector offsets;
  HostIndexVector cols;
  HostVector      wts;
};

FlatControl flatten(const DirichletControl& ctr)
{
  FlatControl out;
  out.offsets.resize(ctr.numStateDofs() + 1);
  for (const DirichletControlMapEntry& entry : ctr.mapEntries())
  {
    ++out.offsets[entry.state_row + 1];
  }
  for (Index i = 0; i < ctr.numStateDofs(); ++i)
  {
    out.offsets[i + 1] += out.offsets[i];
  }

  out.cols.resize(ctr.mapEntries().size());
  out.wts.resize(ctr.mapEntries().size());
  HostIndexVector next = out.offsets;
  for (const DirichletControlMapEntry& entry : ctr.mapEntries())
  {
    const Index k = next[entry.state_row]++;
    out.cols[k]   = entry.ctr_col;
    out.wts[k]    = entry.weight;
  }
  return out;
}

void checkStep(ControlMapView<MemorySpace::Host> map, Index step)
{
  if (step < 0 || step >= map.num_steps)
  {
    throw std::runtime_error("ControlMap step is out of range");
  }
}

void checkControlVectors(ControlMapView<MemorySpace::Host> map,
                         HostConstVectorView               in,
                         HostVectorView                    out,
                         Index                             out_size)
{
  if (in.size() != map.num_prm || out.size() != out_size)
  {
    throw std::runtime_error("ControlMap vector size mismatch");
  }
}

Real ctrVal(ControlMapView<MemorySpace::Host> map,
            Index                             step,
            Index                             i,
            HostConstVectorView               prm)
{
  const Index lo    = map.lower[step];
  const Index hi    = map.upper[step];
  const Real  hi_wt = map.upper_wts[step];
  const Real  lo_wt = 1.0 - hi_wt;
  Real        val   = 0.0;
  for (Index k = map.ctrBegin(i); k < map.ctrEnd(i); ++k)
  {
    const Index col = map.ctr_cols[k];
    Real        q   = lo_wt * prm[map.ctrIndex(lo, col)];
    if (hi != lo && hi_wt != 0.0)
    {
      q += hi_wt * prm[map.ctrIndex(hi, col)];
    }
    val += map.ctr_wts[k] * q;
  }
  return val;
}

void checkInitialVectors(HostConstVectorView in,
                         HostVectorView      out,
                         Index               in_size,
                         Index               out_size)
{
  if (in.size() != in_size || out.size() != out_size)
  {
    throw std::runtime_error("InitialStateMap vector size mismatch");
  }
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
  if (num_steps <= 0 || num_states <= 0 || ctr_off < 0)
  {
    throw std::runtime_error("ControlMap received invalid dimensions");
  }

  HostIndexVector dofs;
  dofs.reserve(ctr.numStateDofs() + fixed_dofs.size());
  Array<char> used(num_states, 0);
  for (Index dof : ctr.stateDofs())
  {
    if (dof < 0 || dof >= num_states || used[dof] != 0)
    {
      throw std::runtime_error("ControlMap controlled DOF is invalid");
    }
    used[dof] = 1;
    dofs.push_back(dof);
  }
  for (Index dof : fixed_dofs)
  {
    if (dof < 0 || dof >= num_states || used[dof] != 0)
    {
      throw std::runtime_error("ControlMap fixed DOF is invalid");
    }
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
  if (time.size() != num_steps)
  {
    throw std::runtime_error("ControlMap time stencil count mismatch");
  }

  HostIndexVector lower(num_steps);
  HostIndexVector upper(num_steps);
  HostVector      upper_wts(num_steps);
  Index           num_levels = 0;
  for (Index step = 0; step < num_steps; ++step)
  {
    if (!time[step].isValid())
    {
      throw std::runtime_error("ControlMap time stencil is invalid");
    }
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
  if (num_prm < required)
  {
    throw std::runtime_error("ControlMap parameter count is too small");
  }

  const Index num_fixed = fixed_dofs.size();
  if (fixed_vals.empty())
  {
    fixed_vals.resize(num_steps * num_fixed);
  }
  else if (fixed_vals.size() == num_fixed)
  {
    HostVector vals(num_steps * num_fixed);
    for (Index step = 0; step < num_steps; ++step)
    {
      for (Index i = 0; i < num_fixed; ++i)
      {
        vals[step * num_fixed + i] = fixed_vals[i];
      }
    }
    fixed_vals = std::move(vals);
  }
  else if (fixed_vals.size() != num_steps * num_fixed)
  {
    throw std::runtime_error("ControlMap fixed value size mismatch");
  }

  FlatControl    flat = flatten(ctr);
  HostControlMap out;
  out.num_steps_   = num_steps;
  out.num_states_  = num_states;
  out.num_prm_     = num_prm;
  out.num_ctr_     = ctr.numStateDofs();
  out.num_fixed_   = num_fixed;
  out.num_ctr_prm_ = ctr.numControlParams();
  out.ctr_off_     = ctr_off;
  out.dofs_        = std::move(dofs);
  out.ctr_offsets_ = std::move(flat.offsets);
  out.ctr_cols_    = std::move(flat.cols);
  out.ctr_wts_     = std::move(flat.wts);
  out.fixed_vals_  = std::move(fixed_vals);
  out.lower_       = std::move(lower);
  out.upper_       = std::move(upper);
  out.upper_wts_   = std::move(upper_wts);
  return out;
}

void copy(const HostControlMap& src,
          DeviceControlMap&     dst,
          CudaContext&          ctx)
{
  dst.num_steps_   = src.num_steps_;
  dst.num_states_  = src.num_states_;
  dst.num_prm_     = src.num_prm_;
  dst.num_ctr_     = src.num_ctr_;
  dst.num_fixed_   = src.num_fixed_;
  dst.num_ctr_prm_ = src.num_ctr_prm_;
  dst.ctr_off_     = src.ctr_off_;
  femx::copy(src.dofs_, dst.dofs_, ctx);
  femx::copy(src.ctr_offsets_, dst.ctr_offsets_, ctx);
  femx::copy(src.ctr_cols_, dst.ctr_cols_, ctx);
  femx::copy(src.ctr_wts_, dst.ctr_wts_, ctx);
  femx::copy(src.fixed_vals_, dst.fixed_vals_, ctx);
  femx::copy(src.lower_, dst.lower_, ctx);
  femx::copy(src.upper_, dst.upper_, ctx);
  femx::copy(src.upper_wts_, dst.upper_wts_, ctx);
}

void controlVals(ControlMapView<MemorySpace::Host> map,
                 Index                             step,
                 HostConstVectorView               prm,
                 HostVectorView                    out)
{
  checkStep(map, step);
  checkControlVectors(map, prm, out, map.numBcs());
  for (Index i = 0; i < map.num_ctr; ++i)
  {
    out[i] = ctrVal(map, step, i, prm);
  }
  for (Index i = 0; i < map.num_fixed; ++i)
  {
    out[map.num_ctr + i] = map.fixedValue(step, i);
  }
}

void controlJac(ControlMapView<MemorySpace::Host> map,
                Index                             step,
                HostConstVectorView               dir,
                HostVectorView                    out)
{
  checkStep(map, step);
  checkControlVectors(map, dir, out, map.num_states);
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = 0.0;
  }
  for (Index i = 0; i < map.num_ctr; ++i)
  {
    out[map.dof(i)] = -ctrVal(map, step, i, dir);
  }
}

void addControlJacT(ControlMapView<MemorySpace::Host> map,
                    Index                             step,
                    HostConstVectorView               adj,
                    HostVectorView                    grad)
{
  checkStep(map, step);
  if (adj.size() != map.num_states || grad.size() != map.num_prm)
  {
    throw std::runtime_error("ControlMap transpose vector size mismatch");
  }

  const Index lo    = map.lower[step];
  const Index hi    = map.upper[step];
  const Real  hi_wt = map.upper_wts[step];
  const Real  lo_wt = 1.0 - hi_wt;
  for (Index i = 0; i < map.num_ctr; ++i)
  {
    const Real val = -adj[map.dof(i)];
    for (Index k = map.ctrBegin(i); k < map.ctrEnd(i); ++k)
    {
      const Index col              = map.ctr_cols[k];
      const Real  wt               = map.ctr_wts[k] * val;
      grad[map.ctrIndex(lo, col)] += lo_wt * wt;
      if (hi != lo && hi_wt != 0.0)
      {
        grad[map.ctrIndex(hi, col)] += hi_wt * wt;
      }
    }
  }
}

HostInitialStateMap makeInitialStateMap(HostVector              mean,
                                        DenseMatrix             modes,
                                        const DirichletControl& ctr,
                                        Index                   init_off,
                                        Index                   ctr_off,
                                        Index                   num_prm)
{
  if (mean.empty() || modes.numRows() != mean.size())
  {
    throw std::runtime_error(
        "InitialStateMap mean and modes must match the state size");
  }
  if (init_off < 0 || ctr_off < 0 || num_prm < 0
      || init_off + modes.numCols() > num_prm
      || ctr_off + ctr.numControlParams() > num_prm)
  {
    throw std::runtime_error(
        "InitialStateMap parameter blocks do not fit the parameter vector");
  }
  for (Index row : ctr.stateDofs())
  {
    if (row < 0 || row >= mean.size())
    {
      throw std::runtime_error(
          "InitialStateMap controlled state DOF is out of range");
    }
    for (Index col = 0; col < modes.numCols(); ++col)
    {
      if (modes(row, col) != 0.0)
      {
        throw std::runtime_error(
            "InitialStateMap modes must vanish on controlled DOFs");
      }
    }
  }

  HostVector flat_modes(modes.size());
  for (Index i = 0; i < modes.size(); ++i)
  {
    flat_modes[i] = modes.data()[i];
  }
  FlatControl flat = flatten(ctr);

  HostInitialStateMap out;
  out.num_states_  = mean.size();
  out.num_prm_     = num_prm;
  out.num_modes_   = modes.numCols();
  out.num_ctr_     = ctr.numStateDofs();
  out.init_off_    = init_off;
  out.ctr_off_     = ctr_off;
  out.mean_        = std::move(mean);
  out.modes_       = std::move(flat_modes);
  out.ctr_dofs_    = ctr.stateDofs();
  out.ctr_offsets_ = std::move(flat.offsets);
  out.ctr_cols_    = std::move(flat.cols);
  out.ctr_wts_     = std::move(flat.wts);
  return out;
}

void copy(const HostInitialStateMap& src,
          DeviceInitialStateMap&     dst,
          CudaContext&               ctx)
{
  dst.num_states_ = src.num_states_;
  dst.num_prm_    = src.num_prm_;
  dst.num_modes_  = src.num_modes_;
  dst.num_ctr_    = src.num_ctr_;
  dst.init_off_   = src.init_off_;
  dst.ctr_off_    = src.ctr_off_;
  femx::copy(src.mean_, dst.mean_, ctx);
  femx::copy(src.modes_, dst.modes_, ctx);
  femx::copy(src.ctr_dofs_, dst.ctr_dofs_, ctx);
  femx::copy(src.ctr_offsets_, dst.ctr_offsets_, ctx);
  femx::copy(src.ctr_cols_, dst.ctr_cols_, ctx);
  femx::copy(src.ctr_wts_, dst.ctr_wts_, ctx);
}

void initialState(InitialStateMapView<MemorySpace::Host> map,
                  HostConstVectorView                    prm,
                  HostVectorView                         out)
{
  checkInitialVectors(prm, out, map.num_prm, map.num_states);
  for (Index row = 0; row < map.num_states; ++row)
  {
    Real val = map.mean[row];
    for (Index col = 0; col < map.num_modes; ++col)
    {
      val += map.mode(row, col) * prm[map.init_off + col];
    }
    out[row] = val;
  }
  for (Index i = 0; i < map.num_ctr; ++i)
  {
    Real val = 0.0;
    for (Index k = map.ctrBegin(i); k < map.ctrEnd(i); ++k)
    {
      val += map.ctr_wts[k] * prm[map.ctr_off + map.ctr_cols[k]];
    }
    out[map.ctr_dofs[i]] = val;
  }
}

void addInitialJacT(InitialStateMapView<MemorySpace::Host> map,
                    HostConstVectorView                    adj,
                    HostVectorView                         grad)
{
  checkInitialVectors(adj, grad, map.num_states, map.num_prm);
  for (Index col = 0; col < map.num_modes; ++col)
  {
    Real val = 0.0;
    for (Index row = 0; row < map.num_states; ++row)
    {
      val += map.mode(row, col) * adj[row];
    }
    grad[map.init_off + col] += val;
  }
  for (Index i = 0; i < map.num_ctr; ++i)
  {
    const Real val = adj[map.ctr_dofs[i]];
    for (Index k = map.ctrBegin(i); k < map.ctrEnd(i); ++k)
    {
      grad[map.ctr_off + map.ctr_cols[k]] += map.ctr_wts[k] * val;
    }
  }
}

#if !defined(FEMX_HAS_CUDA)
void controlVals(ControlMapView<MemorySpace::Device>,
                 Index,
                 DeviceConstVectorView,
                 DeviceVectorView,
                 CudaContext&)
{
  throw std::runtime_error("ControlMap Device operations require CUDA");
}

void controlJac(ControlMapView<MemorySpace::Device>,
                Index,
                DeviceConstVectorView,
                DeviceVectorView,
                CudaContext&)
{
  throw std::runtime_error("ControlMap Device operations require CUDA");
}

void addControlJacT(ControlMapView<MemorySpace::Device>,
                    Index,
                    DeviceConstVectorView,
                    DeviceVectorView,
                    CudaContext&)
{
  throw std::runtime_error("ControlMap Device operations require CUDA");
}

void initialState(InitialStateMapView<MemorySpace::Device>,
                  DeviceConstVectorView,
                  DeviceVectorView,
                  CudaContext&)
{
  throw std::runtime_error("InitialStateMap Device operations require CUDA");
}

void addInitialJacT(InitialStateMapView<MemorySpace::Device>,
                    DeviceConstVectorView,
                    DeviceVectorView,
                    CudaContext&)
{
  throw std::runtime_error("InitialStateMap Device operations require CUDA");
}
#endif

} // namespace fem
} // namespace femx
