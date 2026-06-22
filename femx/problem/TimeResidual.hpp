#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/BlockVectorView.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{
namespace problem
{

struct TimeDims
{
  Index nt   = 0;
  Index nst  = 0;
  Index nprm = 0;
  Index nres = 0;
  Index nhst = 1;
};

/** @brief Non-owning view of time-history states for one residual step. */
class TimeHistoryView
{
public:
  TimeHistoryView() = default;

  TimeHistoryView(const Real* data, Index count, Index nst)
    : data_(data),
      count_(count),
      nst_(nst)
  {
    if (count_ < 0 || nst_ < 0)
    {
      throw std::runtime_error("TimeHistoryView received invalid dimensions");
    }
    if (count_ > 0 && nst_ > 0 && data_ == nullptr)
    {
      throw std::runtime_error("TimeHistoryView received null data");
    }
  }

  bool empty() const
  {
    return count_ == 0;
  }

  Index count() const
  {
    return count_;
  }

  Index stateSize() const
  {
    return nst_;
  }

  VectorView<const Real> state(Index i) const
  {
    if (i < 0 || i >= count_)
    {
      return {};
    }
    return blocks().block(i);
  }

  VectorView<const Real> flat() const
  {
    return blocks().flat();
  }

  const Real* data() const
  {
    return data_;
  }

private:
  BlockVectorView<const Real> blocks() const
  {
    return BlockVectorView<const Real>(data_, count_, nst_);
  }

  const Real* data_{nullptr};
  Index       count_{0};
  Index       nst_{0};
};

/** @brief Context for one residual step.
 *
 * history.state(0) is the state at the current time level for this residual,
 * history.state(1) is one level older, and so on. Missing startup history is
 * supplied by the state solver, typically by clamping to level zero. prev
 * is kept for one-step residuals that still read the legacy field directly.
 */
struct TimeContext
{
  Index               step = 0;
  const Vector<Real>* prev = nullptr;
  const Vector<Real>* nxt  = nullptr;
  const Vector<Real>* prm  = nullptr;
  TimeHistoryView     hist;

  TimeHistoryView historyView() const
  {
    if (!hist.empty())
    {
      return hist;
    }
    if (prev != nullptr)
    {
      return TimeHistoryView(prev->data(), 1, prev->size());
    }
    return {};
  }
};

/** @brief Differentiation block for a time residual.
 *
 * PrevState is kept as an alias for history(0) for one-step schemes and
 * backwards compatibility.
 */
class VariableBlock final
{
public:
  enum class Kind
  {
    HistoryState,
    NextState,
    Param
  };

  constexpr VariableBlock(Kind  kind        = Kind::HistoryState,
                          Index history_lag = 0)
    : kind_(kind),
      history_lag_(history_lag)
  {
  }

  static constexpr VariableBlock hist(Index lag)
  {
    return VariableBlock(Kind::HistoryState, lag);
  }

  bool isHistoryState() const
  {
    return kind_ == Kind::HistoryState;
  }

  bool isNextState() const
  {
    return kind_ == Kind::NextState;
  }

  bool isParam() const
  {
    return kind_ == Kind::Param;
  }

  Kind kind() const
  {
    return kind_;
  }

  Index historyLag() const
  {
    if (!isHistoryState())
    {
      throw std::runtime_error(
          "VariableBlock does not refer to a history state");
    }
    return history_lag_;
  }

  friend bool operator==(VariableBlock lhs, VariableBlock rhs)
  {
    return lhs.kind_ == rhs.kind_ && lhs.history_lag_ == rhs.history_lag_;
  }

  friend bool operator!=(VariableBlock lhs, VariableBlock rhs)
  {
    return !(lhs == rhs);
  }

  static const VariableBlock PrevState;
  static const VariableBlock NextState;
  static const VariableBlock Param;

private:
  Kind  kind_{Kind::HistoryState};
  Index history_lag_{0};
};

inline constexpr VariableBlock VariableBlock::PrevState =
    VariableBlock::hist(0);
inline constexpr VariableBlock VariableBlock::NextState =
    VariableBlock(VariableBlock::Kind::NextState);
inline constexpr VariableBlock VariableBlock::Param =
    VariableBlock(VariableBlock::Kind::Param);

class TimeResidual;

/** @brief Linearization of a TimeResidual at one time-step context. */
class TimeLinearization
{
public:
  void reset(const TimeResidual& problem, TimeContext ctx);

  void applyJac(VariableBlock       wrt,
                const Vector<Real>& dir,
                Vector<Real>&       out) const;

  void applyJacT(VariableBlock       wrt,
                 const Vector<Real>& adj,
                 Vector<Real>&       out) const;

  bool assembleJac(VariableBlock          wrt,
                   linalg::MatrixBuilder& out) const;

private:
  void checkReady() const;

private:
  const TimeResidual* problem_{nullptr};
  TimeContext         ctx_;
};

class TimeResidual
{
public:
  virtual ~TimeResidual() = default;

  virtual TimeDims dims() const = 0;

  virtual void res(const TimeContext& ctx,
                   Vector<Real>&      out) const = 0;

  virtual void applyJac(const TimeContext&  ctx,
                        VariableBlock       wrt,
                        const Vector<Real>& dir,
                        Vector<Real>&       out) const = 0;

  virtual void applyJacT(const TimeContext&  ctx,
                         VariableBlock       wrt,
                         const Vector<Real>& adj,
                         Vector<Real>&       out) const = 0;

  virtual void linearize(const TimeContext& ctx,
                         TimeLinearization& out) const
  {
    out.reset(*this, ctx);
  }

  virtual bool assembleJac(const TimeContext&     ctx,
                           VariableBlock          wrt,
                           linalg::MatrixBuilder& out) const
  {
    (void) ctx;
    (void) wrt;
    (void) out;
    return false;
  }

  virtual void prepareLinearSolve(const TimeContext&     ctx,
                                  VariableBlock          wrt,
                                  linalg::MatrixBuilder& jac,
                                  Vector<Real>&          rhs) const
  {
    (void) ctx;
    (void) wrt;
    (void) jac;
    (void) rhs;
  }
};

inline void TimeLinearization::reset(const TimeResidual& problem,
                                     TimeContext         ctx)
{
  problem_ = &problem;
  ctx_     = ctx;
}

inline void TimeLinearization::applyJac(VariableBlock       wrt,
                                        const Vector<Real>& dir,
                                        Vector<Real>&       out) const
{
  checkReady();
  problem_->applyJac(ctx_, wrt, dir, out);
}

inline void TimeLinearization::applyJacT(VariableBlock       wrt,
                                         const Vector<Real>& adj,
                                         Vector<Real>&       out) const
{
  checkReady();
  problem_->applyJacT(ctx_, wrt, adj, out);
}

inline bool TimeLinearization::assembleJac(
    VariableBlock          wrt,
    linalg::MatrixBuilder& out) const
{
  checkReady();
  return problem_->assembleJac(ctx_, wrt, out);
}

inline void TimeLinearization::checkReady() const
{
  if (problem_ == nullptr)
  {
    throw std::runtime_error(
        "TimeLinearization used before TimeResidual::linearize");
  }
}

} // namespace problem
} // namespace femx
