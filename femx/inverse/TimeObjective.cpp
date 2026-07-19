#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "TimeObjectivePlan.hpp"
#include <femx/common/Checks.hpp>
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

#if !defined(FEMX_HAS_CUDA)
void noCuda()
{
  throw std::runtime_error(
      "Device objective execution requires FEMX_ENABLE_CUDA");
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

} // namespace

void TimeObjectivePlan::add(const TimeObjective& obj, CudaContext& ctx)
{
  const Index first_ls = ls_.size();
  setDims(obj);
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

Index TimeObjectivePlan::numSteps() const noexcept
{
  return num_steps_ < 0 ? 0 : num_steps_;
}

Index TimeObjectivePlan::numStates() const noexcept
{
  return num_states_ < 0 ? 0 : num_states_;
}

Index TimeObjectivePlan::numParams() const noexcept
{
  return num_prm_ < 0 ? 0 : num_prm_;
}

Real TimeObjectivePlan::value(const state::DeviceTimeTrajectory& tr,
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

  if (q_terms_ > 0)
  {
    apply(q_mat_, prm, q_prod_.view(), ctx);
    dot(prm, q_prod_.view(), q_dot_.view(), ctx);
    axpby(0.5, q_dot_.view(), 1.0, scalar_.view(), ctx);
    dot(prm, q_lin_.view(), q_dot_.view(), ctx);
    axpby(1.0, q_dot_.view(), 1.0, scalar_.view(), ctx);
  }

  copy(scalar_, scalar_h_, ctx);
  ctx.synchronize();
  return scalar_h_[0] + q_const_;
}

void TimeObjectivePlan::stateGrad(
    Index                              level,
    const state::DeviceTimeTrajectory& tr,
    DeviceConstVectorView              prm,
    DeviceVectorView                   out,
    CudaContext&                       ctx) const
{
  checkInputs(tr, prm);
  checkLevel(level);
  require(out.size() == numStates(),
          "Device objective state gradient size mismatch");
  zero(out, ctx);

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

void TimeObjectivePlan::paramGrad(
    const state::DeviceTimeTrajectory& tr,
    DeviceConstVectorView              prm,
    DeviceVectorView                   out,
    CudaContext&                       ctx) const
{
  checkInputs(tr, prm);
  require(out.size() == numParams(),
          "Device objective parameter gradient size mismatch");
  zero(out, ctx);
  if (q_terms_ > 0)
  {
    apply(q_mat_, prm, out, ctx);
    axpby(1.0, q_lin_.view(), 1.0, out, ctx);
  }
}

void TimeObjectivePlan::flatten(const TimeObjective& obj,
                                CudaContext&         ctx)
{
  if (const auto* sum = dynamic_cast<const SumTimeObjective*>(&obj))
  {
    for (const TimeObjective* term : sum->terms())
    {
      setDims(*term);
      flatten(*term, ctx);
    }
    return;
  }
  if (const auto* ls =
          dynamic_cast<const TimeLeastSquaresObjective*>(&obj))
  {
    auto obs = ls->obs_.copyToDevice(ctx);
    require(obs != nullptr,
            "Device objective observation has no Device operator");
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
      "Device objective plan does not support this objective type");
}

void TimeObjectivePlan::addLeastSquares(
    const TimeLeastSquaresObjective&               obj,
    std::unique_ptr<DeviceTimeObservationOperator> obs)
{
  require(obs->numSteps() == numSteps() && obs->numStates() == numStates()
              && obs->numObservations() == obj.data_.numObservations(),
          "Device objective observation size mismatch");

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

void TimeObjectivePlan::uploadLeastSquares(Index        first,
                                           CudaContext& ctx)
{
  for (Index i = first; i < ls_.size(); ++i)
  {
    LeastSquaresTerm& term = ls_[i];
    copy(term.data_h, term.data, ctx);
    copy(term.obs_wts_h, term.obs_wts, ctx);
  }
}

void TimeObjectivePlan::addRegularization(
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

void TimeObjectivePlan::addRegularization(
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

void TimeObjectivePlan::appendQuadratic(Index row,
                                        Index col,
                                        Real  val,
                                        Real  row_ref,
                                        Real  col_ref)
{
  const Real half = 0.5 * val;
  q_rows_h_.push_back(row);
  q_cols_h_.push_back(col);
  q_vals_h_.push_back(half);
  q_rows_h_.push_back(col);
  q_cols_h_.push_back(row);
  q_vals_h_.push_back(half);

  q_lin_h_[row] -= half * col_ref;
  q_lin_h_[col] -= half * row_ref;
  q_const_      += half * row_ref * col_ref;
  ++q_terms_;
}

void TimeObjectivePlan::setDims(const TimeObjective& obj)
{
  if (num_steps_ < 0)
  {
    num_steps_  = obj.numSteps();
    num_states_ = obj.numStates();
    num_prm_    = obj.numParams();
    scalar_.resize(1);
    scalar_h_.resize(1);
    q_lin_h_.resize(num_prm_);
    return;
  }
  require(obj.numSteps() == numSteps() && obj.numStates() == numStates()
              && obj.numParams() == numParams(),
          "Device objective plan received inconsistent dimensions");
}

void TimeObjectivePlan::uploadQuadratic(CudaContext& ctx)
{
  if (q_terms_ == 0)
  {
    return;
  }

  struct Entry
  {
    Index row;
    Index col;
    Real  val;
  };

  Array<Entry> entries;
  entries.reserve(q_vals_h_.size());
  for (Index i = 0; i < q_vals_h_.size(); ++i)
  {
    entries.push_back({q_rows_h_[i], q_cols_h_[i], q_vals_h_[i]});
  }
  std::sort(entries.begin(),
            entries.end(),
            [](const Entry& lhs, const Entry& rhs)
            {
              return std::tie(lhs.row, lhs.col)
                     < std::tie(rhs.row, rhs.col);
            });

  HostIndexVector row_ptr(numParams() + 1);
  HostIndexVector col_ind;
  HostVector      vals;
  for (Index i = 0; i < entries.size();)
  {
    const Index row = entries[i].row;
    const Index col = entries[i].col;
    Real        val = 0.0;
    do
    {
      val += entries[i].val;
      ++i;
    } while (i < entries.size() && entries[i].row == row
             && entries[i].col == col);
    if (val != 0.0)
    {
      ++row_ptr[row + 1];
      col_ind.push_back(col);
      vals.push_back(val);
    }
  }
  for (Index row = 0; row < numParams(); ++row)
  {
    row_ptr[row + 1] += row_ptr[row];
  }

  HostCsrGraph  host_graph(numParams(),
                          numParams(),
                          std::move(row_ptr),
                          std::move(col_ind));
  HostCsrMatrix host_mat(host_graph);
  host_mat.vals() = std::move(vals);
  DeviceCsrGraph graph;
  copy(host_graph, graph, ctx);
  DeviceCsrMatrix mat(graph);
  copy(host_mat, mat, ctx);
  q_mat_ = std::move(mat);
  copy(q_lin_h_, q_lin_, ctx);
  q_prod_.resize(numParams());
  q_dot_.resize(1);
}

void TimeObjectivePlan::checkInputs(
    const state::DeviceTimeTrajectory& tr,
    DeviceConstVectorView              prm) const
{
  require(num_steps_ >= 0, "Device objective plan is empty");
  require(tr.numSteps() == numSteps() && tr.numStates() == numStates()
              && prm.size() == numParams(),
          "Device objective input size mismatch");
}

void TimeObjectivePlan::checkLevel(Index level) const
{
  require(level >= 0 && level <= numSteps(),
          "Device objective time level is out of range");
}

} // namespace inverse
} // namespace femx
