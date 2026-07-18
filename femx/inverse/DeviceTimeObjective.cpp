#include <stdexcept>
#include <utility>

#include <femx/inverse/DeviceTimeObjective.hpp>
#include <femx/inverse/SumTimeObjective.hpp>
#include <femx/inverse/TimeBlockRegularization.hpp>
#include <femx/inverse/TimeLeastSquaresObjective.hpp>
#include <femx/inverse/TimeRegularization.hpp>

namespace femx
{
namespace inverse
{
namespace detail
{

void launchLsValue(DeviceConstVectorView lo,
                   DeviceConstVectorView hi,
                   DeviceConstVectorView data,
                   DeviceConstVectorView wts,
                   Real                  lo_wt,
                   Real                  hi_wt,
                   Real                  row_wt,
                   DeviceVectorView      scalar,
                   CudaContext&          ctx);

void launchLsDir(DeviceConstVectorView lo,
                 DeviceConstVectorView hi,
                 DeviceConstVectorView data,
                 DeviceConstVectorView wts,
                 Real                  lo_wt,
                 Real                  hi_wt,
                 Real                  scale,
                 DeviceVectorView      dir,
                 CudaContext&          ctx);

using DeviceConstIndexView =
    VectorView<MemorySpace::Device, const Index>;

void launchQuadraticValue(DeviceConstIndexView  rows,
                          DeviceConstIndexView  cols,
                          DeviceConstVectorView vals,
                          DeviceConstVectorView row_ref,
                          DeviceConstVectorView col_ref,
                          DeviceConstVectorView prm,
                          DeviceVectorView      scalar,
                          CudaContext&          ctx);

void launchQuadraticGrad(DeviceConstIndexView  rows,
                         DeviceConstIndexView  cols,
                         DeviceConstVectorView vals,
                         DeviceConstVectorView row_ref,
                         DeviceConstVectorView col_ref,
                         DeviceConstVectorView prm,
                         DeviceVectorView      out,
                         CudaContext&          ctx);

#if !defined(FEMX_HAS_CUDA)
void noCuda()
{
  throw std::runtime_error(
      "DeviceTimeObjective requires FEMX_ENABLE_CUDA");
}

void launchLsValue(DeviceConstVectorView,
                   DeviceConstVectorView,
                   DeviceConstVectorView,
                   DeviceConstVectorView,
                   Real,
                   Real,
                   Real,
                   DeviceVectorView,
                   CudaContext&)
{
  noCuda();
}

void launchLsDir(DeviceConstVectorView,
                 DeviceConstVectorView,
                 DeviceConstVectorView,
                 DeviceConstVectorView,
                 Real,
                 Real,
                 Real,
                 DeviceVectorView,
                 CudaContext&)
{
  noCuda();
}

void launchQuadraticValue(DeviceConstIndexView,
                          DeviceConstIndexView,
                          DeviceConstVectorView,
                          DeviceConstVectorView,
                          DeviceConstVectorView,
                          DeviceConstVectorView,
                          DeviceVectorView,
                          CudaContext&)
{
  noCuda();
}

void launchQuadraticGrad(DeviceConstIndexView,
                         DeviceConstIndexView,
                         DeviceConstVectorView,
                         DeviceConstVectorView,
                         DeviceConstVectorView,
                         DeviceConstVectorView,
                         DeviceVectorView,
                         CudaContext&)
{
  noCuda();
}
#endif

} // namespace detail

namespace
{

DeviceConstVectorView prefix(const DeviceVector& vec, Index size)
{
  return {vec.data(), size};
}

DeviceVectorView prefix(DeviceVector& vec, Index size)
{
  return {vec.data(), size};
}

DeviceConstVectorView row(const DeviceVector& vec,
                          Index               i,
                          Index               size)
{
  return {vec.data() + i * size, size};
}

detail::DeviceConstIndexView indices(
    const DeviceIndexVector& vec)
{
  return vec.view();
}

} // namespace

void DeviceTimeObjective::add(const TimeObjective& obj, CudaContext& ctx)
{
  const Index first_ls = ls_.size();
  setDimensions(obj);
  flatten(obj, ctx);
  uploadLeastSquares(first_ls, ctx);
  uploadQuadratic(ctx);
  ctx.synchronize();
  for (Index i = first_ls; i < ls_.size(); ++i)
  {
    ls_[i].data_h    = HostVector{};
    ls_[i].obs_wts_h = HostVector{};
  }
}

Index DeviceTimeObjective::numSteps() const noexcept
{
  return num_steps_ < 0 ? 0 : num_steps_;
}

Index DeviceTimeObjective::numStates() const noexcept
{
  return num_states_ < 0 ? 0 : num_states_;
}

Index DeviceTimeObjective::numParams() const noexcept
{
  return num_prm_ < 0 ? 0 : num_prm_;
}

Real DeviceTimeObjective::value(const state::DeviceTimeTrajectory& tr,
                                DeviceConstVectorView              prm,
                                CudaContext&                       ctx) const
{
  checkInputs(tr, prm);
  scalar_.setZero(ctx);

  for (const LeastSquaresTerm& term : ls_)
  {
    const DeviceVectorView lo = prefix(lo_, term.num_obs);
    const DeviceVectorView hi = prefix(hi_, term.num_obs);
    for (Index ir = 0; ir < term.interp.size(); ++ir)
    {
      const LinearInterpolation ip = term.interp[ir];
      term.obs->observe(ip.lower, tr[ip.lower], lo, ctx);
      if (ip.hasUpper())
      {
        term.obs->observe(ip.upper, tr[ip.upper], hi, ctx);
      }
      detail::launchLsValue(
          lo,
          hi,
          row(term.data, ir, term.num_obs),
          row(term.obs_wts, ir, term.num_obs),
          ip.lowerWeight(),
          ip.upperWeight(),
          term.row_wts[ir],
          scalar_.view(),
          ctx);
    }
  }

  if (!q_vals_.empty())
  {
    detail::launchQuadraticValue(
        indices(q_rows_),
        indices(q_cols_),
        q_vals_.view(),
        q_row_ref_.view(),
        q_col_ref_.view(),
        prm,
        scalar_.view(),
        ctx);
  }

  copy(scalar_, scalar_h_, ctx);
  ctx.synchronize();
  return scalar_h_[0];
}

void DeviceTimeObjective::stateGrad(
    Index                              level,
    const state::DeviceTimeTrajectory& tr,
    DeviceConstVectorView              prm,
    DeviceVectorView                   out,
    CudaContext&                       ctx) const
{
  checkInputs(tr, prm);
  checkLevel(level);
  if (out.size() != numStates())
  {
    throw std::runtime_error(
        "DeviceTimeObjective state gradient size mismatch");
  }
  if (!out.empty())
  {
    device::zero(out.data(),
                 static_cast<std::size_t>(out.size()) * sizeof(Real),
                 ctx.stream());
  }

  for (const LeastSquaresTerm& term : ls_)
  {
    const DeviceVectorView lo  = prefix(lo_, term.num_obs);
    const DeviceVectorView hi  = prefix(hi_, term.num_obs);
    const DeviceVectorView dir = prefix(dir_, term.num_obs);
    for (Index ir = 0; ir < term.interp.size(); ++ir)
    {
      const LinearInterpolation ip = term.interp[ir];
      Real                      wt = 0.0;
      if (ip.lower == level)
      {
        wt += ip.lowerWeight();
      }
      if (ip.hasUpper() && ip.upper == level)
      {
        wt += ip.upperWeight();
      }
      wt *= term.row_wts[ir];
      if (wt == 0.0)
      {
        continue;
      }

      term.obs->observe(ip.lower, tr[ip.lower], lo, ctx);
      if (ip.hasUpper())
      {
        term.obs->observe(ip.upper, tr[ip.upper], hi, ctx);
      }
      detail::launchLsDir(
          lo,
          hi,
          row(term.data, ir, term.num_obs),
          row(term.obs_wts, ir, term.num_obs),
          ip.lowerWeight(),
          ip.upperWeight(),
          wt,
          dir,
          ctx);
      term.obs->addStateJacT(level, dir, out, ctx);
    }
  }
}

void DeviceTimeObjective::paramGrad(
    const state::DeviceTimeTrajectory& tr,
    DeviceConstVectorView              prm,
    DeviceVectorView                   out,
    CudaContext&                       ctx) const
{
  checkInputs(tr, prm);
  if (out.size() != numParams())
  {
    throw std::runtime_error(
        "DeviceTimeObjective parameter gradient size mismatch");
  }
  if (!out.empty())
  {
    device::zero(out.data(),
                 static_cast<std::size_t>(out.size()) * sizeof(Real),
                 ctx.stream());
  }
  if (!q_vals_.empty())
  {
    detail::launchQuadraticGrad(
        indices(q_rows_),
        indices(q_cols_),
        q_vals_.view(),
        q_row_ref_.view(),
        q_col_ref_.view(),
        prm,
        out,
        ctx);
  }
}

void DeviceTimeObjective::flatten(const TimeObjective& obj,
                                  CudaContext&         ctx)
{
  if (const auto* sum = dynamic_cast<const SumTimeObjective*>(&obj))
  {
    for (const TimeObjective* term : sum->terms())
    {
      setDimensions(*term);
      flatten(*term, ctx);
    }
    return;
  }
  if (const auto* ls =
          dynamic_cast<const TimeLeastSquaresObjective*>(&obj))
  {
    auto obs = ls->obs_.copyToDevice(ctx);
    if (!obs)
    {
      throw std::runtime_error(
          "DeviceTimeObjective observation has no Device operator");
    }
    addLeastSquares(*ls, std::move(obs));
    return;
  }
  if (const auto* reg = dynamic_cast<const TimeRegularization*>(&obj))
  {
    addRegularization(*reg);
    return;
  }
  if (const auto* reg =
          dynamic_cast<const TimeBlockRegularization*>(&obj))
  {
    addRegularization(*reg);
    return;
  }
  throw std::runtime_error(
      "DeviceTimeObjective does not support this objective type");
}

void DeviceTimeObjective::addLeastSquares(
    const TimeLeastSquaresObjective&               obj,
    std::unique_ptr<DeviceTimeObservationOperator> obs)
{
  if (obs->numSteps() != numSteps() || obs->numStates() != numStates()
      || obs->numObservations() != obj.data_.numObservations())
  {
    throw std::runtime_error(
        "DeviceTimeObjective observation size mismatch");
  }

  LeastSquaresTerm term;
  term.obs     = std::move(obs);
  term.num_obs = obj.data_.numObservations();
  term.interp.resize(obj.data_.numTimeLevels());
  term.row_wts.resize(obj.data_.numTimeLevels());
  term.data_h.resize(obj.data_.numTimeLevels() * term.num_obs);
  term.obs_wts_h = obj.obs_wts_;

  for (Index ir = 0; ir < obj.data_.numTimeLevels(); ++ir)
  {
    term.interp[ir]                = obj.interpolation(ir);
    term.row_wts[ir]               = obj.observationWeight(term.interp[ir]);
    const HostConstVectorView vals = obj.data_[ir];
    for (Index i = 0; i < term.num_obs; ++i)
    {
      term.data_h[ir * term.num_obs + i] = vals[i];
    }
  }

  ls_.push_back(std::move(term));

  const Index n = ls_.back().obs->numObservations();
  if (lo_.size() < n)
  {
    lo_.resize(n);
    hi_.resize(n);
    dir_.resize(n);
  }
}

void DeviceTimeObjective::uploadLeastSquares(Index        first,
                                             CudaContext& ctx)
{
  for (Index i = first; i < ls_.size(); ++i)
  {
    LeastSquaresTerm& term = ls_[i];
    copy(term.data_h, term.data, ctx);
    copy(term.obs_wts_h, term.obs_wts, ctx);
  }
}

void DeviceTimeObjective::addRegularization(
    const TimeRegularization& obj)
{
  if (obj.beta_value_ != 0.0)
  {
    for (Index i = 0; i < obj.numParams(); ++i)
    {
      appendQuadratic(i,
                      i,
                      obj.beta_value_,
                      obj.reference_[i],
                      obj.reference_[i]);
    }
  }

  if (obj.beta_diff_ == 0.0)
  {
    return;
  }
  for (Index level = 1; level < obj.num_levels_; ++level)
  {
    for (Index c = 0; c < obj.block_size_; ++c)
    {
      const Index curr = obj.index(level, c);
      const Index prev = obj.index(level - 1, c);
      appendQuadratic(curr,
                      curr,
                      obj.beta_diff_,
                      obj.reference_[curr],
                      obj.reference_[curr]);
      appendQuadratic(prev,
                      prev,
                      obj.beta_diff_,
                      obj.reference_[prev],
                      obj.reference_[prev]);
      appendQuadratic(curr,
                      prev,
                      -obj.beta_diff_,
                      obj.reference_[curr],
                      obj.reference_[prev]);
      appendQuadratic(prev,
                      curr,
                      -obj.beta_diff_,
                      obj.reference_[prev],
                      obj.reference_[curr]);
    }
  }
}

void DeviceTimeObjective::addRegularization(
    const TimeBlockRegularization& obj)
{
  if (obj.weight_ == 0.0)
  {
    return;
  }
  for (Index level = 0; level < obj.num_levels_; ++level)
  {
    for (Index i = 0; i < obj.vals_.size(); ++i)
    {
      const Index row = obj.index(level, obj.rows_[i]);
      const Index col = obj.index(level, obj.cols_[i]);
      appendQuadratic(row,
                      col,
                      obj.weight_ * obj.vals_[i],
                      obj.reference_[row],
                      obj.reference_[col]);
    }
  }
}

void DeviceTimeObjective::appendQuadratic(Index row,
                                          Index col,
                                          Real  val,
                                          Real  row_ref,
                                          Real  col_ref)
{
  q_rows_h_.push_back(row);
  q_cols_h_.push_back(col);
  q_vals_h_.push_back(val);
  q_row_ref_h_.push_back(row_ref);
  q_col_ref_h_.push_back(col_ref);
}

void DeviceTimeObjective::setDimensions(const TimeObjective& obj)
{
  if (num_steps_ < 0)
  {
    num_steps_  = obj.numSteps();
    num_states_ = obj.numStates();
    num_prm_    = obj.numParams();
    scalar_.resize(1);
    scalar_h_.resize(1);
    return;
  }
  if (obj.numSteps() != numSteps() || obj.numStates() != numStates()
      || obj.numParams() != numParams())
  {
    throw std::runtime_error(
        "DeviceTimeObjective received inconsistent dimensions");
  }
}

void DeviceTimeObjective::uploadQuadratic(CudaContext& ctx)
{
  copy(q_rows_h_, q_rows_, ctx);
  copy(q_cols_h_, q_cols_, ctx);
  copy(q_vals_h_, q_vals_, ctx);
  copy(q_row_ref_h_, q_row_ref_, ctx);
  copy(q_col_ref_h_, q_col_ref_, ctx);
}

void DeviceTimeObjective::checkInputs(
    const state::DeviceTimeTrajectory& tr,
    DeviceConstVectorView              prm) const
{
  if (num_steps_ < 0)
  {
    throw std::runtime_error("DeviceTimeObjective is empty");
  }
  if (tr.numSteps() != numSteps() || tr.numStates() != numStates()
      || prm.size() != numParams())
  {
    throw std::runtime_error("DeviceTimeObjective input size mismatch");
  }
}

void DeviceTimeObjective::checkLevel(Index level) const
{
  if (level < 0 || level > numSteps())
  {
    throw std::runtime_error(
        "DeviceTimeObjective time level is out of range");
  }
}

} // namespace inverse
} // namespace femx
