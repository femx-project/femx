#pragma once

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/SystemMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/View.hpp>

namespace femx::state
{

/** @brief Dimensions of a time-dependent residual problem. */
struct TimeDims
{
  Index num_steps  = 0;
  Index num_states = 0;
  Index num_param  = 0;
  Index num_res    = 0;
  Index num_hist   = 1;
};

/** @brief Non-owning lag-major history window. */
template <MemorySpace Space>
class TimeHistoryView
{
public:
  FEMX_HOST_DEVICE TimeHistoryView() = default;

  FEMX_HOST_DEVICE TimeHistoryView(const Real* data,
                                   Index       count,
                                   Index       num_states)
    : data_(data), count_(count), num_states_(num_states)
  {
  }

  FEMX_HOST_DEVICE bool empty() const
  {
    return count_ == 0;
  }

  FEMX_HOST_DEVICE Index count() const
  {
    return count_;
  }

  FEMX_HOST_DEVICE Index stateSize() const
  {
    return num_states_;
  }

  FEMX_HOST_DEVICE VectorView<Space, const Real> state(Index i) const
  {
    return blocks().block(i);
  }

  FEMX_HOST_DEVICE const Real* data() const
  {
    return data_;
  }

  FEMX_HOST_DEVICE VectorView<Space, const Real> values() const
  {
    return {data_, count_ * num_states_};
  }

private:
  FEMX_HOST_DEVICE BlockVectorView<Space, const Real> blocks() const
  {
    return {data_, count_, num_states_};
  }

  const Real* data_{nullptr};
  Index       count_{0};
  Index       num_states_{0};
};

/** @brief Native-memory inputs at one residual step. */
template <MemorySpace Space>
struct TimeContext
{
  Index                         step{0};
  VectorView<Space, const Real> nxt;
  VectorView<Space, const Real> prm;
  TimeHistoryView<Space>        hist;
};

using HostTimeHistoryView   = TimeHistoryView<MemorySpace::Host>;
using DeviceTimeHistoryView = TimeHistoryView<MemorySpace::Device>;
using HostTimeContext       = TimeContext<MemorySpace::Host>;
using DeviceTimeContext     = TimeContext<MemorySpace::Device>;

/** @brief Variable block of a time-residual Jacobian. */
class VariableBlock final
{
public:
  enum class Kind
  {
    HistoryState,
    NextState,
    Param
  };

  FEMX_HOST_DEVICE constexpr VariableBlock(
      Kind  kind     = Kind::HistoryState,
      Index hist_lag = 0)
    : kind_(kind), hist_lag_(hist_lag)
  {
  }

  FEMX_HOST_DEVICE static constexpr VariableBlock hist(Index lag)
  {
    return {Kind::HistoryState, lag};
  }

  FEMX_HOST_DEVICE constexpr bool isHistoryState() const
  {
    return kind_ == Kind::HistoryState;
  }

  FEMX_HOST_DEVICE constexpr bool isNextState() const
  {
    return kind_ == Kind::NextState;
  }

  FEMX_HOST_DEVICE constexpr bool isParam() const
  {
    return kind_ == Kind::Param;
  }

  FEMX_HOST_DEVICE constexpr Kind kind() const
  {
    return kind_;
  }

  FEMX_HOST_DEVICE constexpr Index historyLag() const
  {
    return hist_lag_;
  }

  FEMX_HOST_DEVICE friend constexpr bool operator==(
      VariableBlock lhs,
      VariableBlock rhs)
  {
    return lhs.kind_ == rhs.kind_ && lhs.hist_lag_ == rhs.hist_lag_;
  }

  FEMX_HOST_DEVICE friend constexpr bool operator!=(
      VariableBlock lhs,
      VariableBlock rhs)
  {
    return !(lhs == rhs);
  }

  static const VariableBlock NextState;
  static const VariableBlock Param;

private:
  Kind  kind_{Kind::HistoryState};
  Index hist_lag_{0};
};

inline constexpr VariableBlock VariableBlock::NextState{
    VariableBlock::Kind::NextState};
inline constexpr VariableBlock VariableBlock::Param{
    VariableBlock::Kind::Param};

template <MemorySpace Space>
class TimeResidual;

/** @brief Define a time residual in one memory space. */
template <MemorySpace Space>
class TimeResidual
{
public:
  static constexpr MemorySpace space = Space;

  using Vec       = Vector<Space, Real>;
  using VecView   = VectorView<Space, Real>;
  using ConstView = VectorView<Space, const Real>;
  using Jac       = linalg::SystemMatrix<Space>;
  using Ctx       = linalg::Context<space>;
  using StepCtx   = TimeContext<space>;

  virtual ~TimeResidual() = default;

  virtual TimeDims dims() const = 0;

  /** @brief Return the canonical Host Jacobian pattern. */
  virtual const HostCsrPattern& hostPattern() const = 0;

  /** @brief Evaluate the parameter-dependent initial state. */
  virtual void initialState(ConstView prm, Vec& out, Ctx& ctx) const = 0;

  /** @brief Add the initial-state Jacobian transpose product to `out`. */
  virtual void addInitialStateJacT(ConstView state_grad,
                                   VecView   out,
                                   Ctx&) const
  {
    const TimeDims dim = dims();
    require(state_grad.size() == dim.num_states
                && out.size() == dim.num_param,
            "TimeResidual initial-state transpose size mismatch");
  }

  /** @brief Assemble the residual and next-state Jacobian together. */
  virtual void assembleNext(const StepCtx& time,
                            Vec&           res,
                            Jac&           jac,
                            Ctx&           ctx) const = 0;

  /** @brief Apply a history or parameter Jacobian transpose. */
  virtual void applyJacT(const StepCtx& time,
                         VariableBlock  wrt,
                         ConstView      adj,
                         Vec&           out,
                         Ctx&           ctx) const = 0;

  /**
   * @brief Set up the Jacobian and right-hand side before a linear solve.
   *
   * @param[in] time - Current time-step context.
   * @param[in,out] jac - Assembled Jacobian.
   * @param[in,out] rhs - Linear-system right-hand side.
   * @param[in,out] ctx - Linear algebra context.
   */
  virtual void setup(const StepCtx& time,
                     Jac&           jac,
                     Vec&           rhs,
                     Ctx&           ctx) const
  {
    (void) time;
    (void) jac;
    (void) rhs;
    (void) ctx;
  }
};

using HostTimeResidual   = TimeResidual<MemorySpace::Host>;
using DeviceTimeResidual = TimeResidual<MemorySpace::Device>;

} // namespace femx::state
