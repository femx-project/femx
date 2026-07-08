#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/BlockVectorView.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>
#include <femx/linalg/MatrixBuilder.hpp>

namespace femx
{
namespace problem
{

/**
 * @brief Sizes that define a time-dependent residual problem.
 *
 * A trajectory has num_steps + 1 state levels. Residual step t advances the
 * history window ending at level t to the next state at level t + 1.
 */
struct TimeDims
{
  Index num_steps          = 0; ///< Number of residual steps.
  Index num_states         = 0; ///< Size of one state vector.
  Index num_params         = 0; ///< Size of the parameter vector shared by all steps.
  Index num_residuals      = 0; ///< Size of one residual vector.
  Index num_history_states = 1; ///< Number of history states required by each step.
};

/**
 * @brief Non-owning view of time-history states for one residual step.
 *
 * TimeHistoryView presents contiguous history data as indexed state vectors
 * without copying the underlying storage.
 */
class TimeHistoryView
{
public:
  TimeHistoryView() = default;

  TimeHistoryView(const Real* data, Index count, Index num_states)
    : data_(data),
      count_(count),
      num_states_(num_states)
  {
    if (count_ < 0 || num_states_ < 0)
    {
      throw std::runtime_error("TimeHistoryView received invalid dimensions");
    }
    if (count_ > 0 && num_states_ > 0 && data_ == nullptr)
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
    return num_states_;
  }

  VectorView<const Real> state(Index i) const
  {
    if (i < 0 || i >= count_)
    {
      throw std::runtime_error("TimeHistoryView state index is out of range");
    }
    return blocks().block(i);
  }

  const Real* data() const
  {
    return data_;
  }

private:
  BlockVectorView<const Real> blocks() const
  {
    return BlockVectorView<const Real>(data_, count_, num_states_);
  }

  const Real* data_{nullptr};
  Index       count_{0};
  Index       num_states_{0};
};

/**
 * @brief Context for one residual step.
 *
 * history.state(0) is the state at the current time level for this residual,
 * history.state(1) is one level older.
 */
struct TimeContext
{
  Index               step = 0;
  const Vector<Real>* nxt  = nullptr;
  const Vector<Real>* prm  = nullptr;
  TimeHistoryView     hist;
};

/**
 * @brief Differentiation block for a time residual.
 *
 * History states are addressed by lag, with hist(0) referring to the current
 * time level for the residual.
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
      hist_lag_(history_lag)
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
    return hist_lag_;
  }

  friend bool operator==(VariableBlock lhs, VariableBlock rhs)
  {
    return lhs.kind_ == rhs.kind_ && lhs.hist_lag_ == rhs.hist_lag_;
  }

  friend bool operator!=(VariableBlock lhs, VariableBlock rhs)
  {
    return !(lhs == rhs);
  }

  static const VariableBlock NextState;
  static const VariableBlock Param;

private:
  Kind  kind_{Kind::HistoryState};
  Index hist_lag_{0};
};

inline constexpr VariableBlock VariableBlock::NextState = VariableBlock(VariableBlock::Kind::NextState);
inline constexpr VariableBlock VariableBlock::Param     = VariableBlock(VariableBlock::Kind::Param);

class TimeResidual;

/**
 * @brief Linearization of a TimeResidual at one time-step context.
 *
 * TimeLinearization binds a residual and context so callers can apply or
 * assemble Jacobians for individual variable blocks.
 */
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

  /**
   * @brief Return the dimensions and history depth expected by this residual.
   */
  virtual TimeDims dims() const = 0;

  /**
   * @brief Evaluate the residual for one time step.
   *
   * @param ctx Step context containing history states, next state, and the parameter vector.
   * @param out Output residual. Implementations may resize it to dims().num_residuals.
   * @throws std::runtime_error if ctx has inconsistent sizes.
   */
  virtual void res(const TimeContext& ctx,
                   Vector<Real>&      out) const = 0;

  /**
   * @brief Apply a Jacobian block to a direction vector.
   *
   * @param ctx Step context used for the linearization point.
   * @param wrt Variable block to differentiate with respect to.
   * @param dir Direction vector with the size of wrt.
   * @param out Output vector of size dims().num_residuals.
   */
  virtual void applyJac(const TimeContext&  ctx,
                        VariableBlock       wrt,
                        const Vector<Real>& dir,
                        Vector<Real>&       out) const = 0;

  /**
   * @brief Apply the transpose of a Jacobian block to an adjoint vector.
   *
   * @param ctx Step context used for the linearization point.
   * @param wrt Variable block to differentiate with respect to.
   * @param adj Adjoint vector of size dims().num_residuals.
   * @param out Output vector with the size of wrt.
   */
  virtual void applyJacT(const TimeContext&  ctx,
                         VariableBlock       wrt,
                         const Vector<Real>& adj,
                         Vector<Real>&       out) const = 0;

  /**
   * @brief Store a lightweight linearization object for repeated applies.
   */
  virtual void linearize(const TimeContext& ctx,
                         TimeLinearization& out) const
  {
    out.reset(*this, ctx);
  }

  /**
   * @brief Assemble a Jacobian block when this residual supports it.
   *
   * @return true if out was filled, false if the block is only available
   *         through applyJac/applyJacT.
   */
  virtual bool assembleJac(const TimeContext&     ctx,
                           VariableBlock          wrt,
                           linalg::MatrixBuilder& out) const
  {
    (void) ctx;
    (void) wrt;
    (void) out;
    return false;
  }

  /**
   * @brief Optional hook to modify an assembled system before a linear solve.
   *
   * Residual wrappers use this to apply constraints that are easier to express
   * on the assembled matrix and right-hand side than in the local Jacobian.
   */
  virtual void prepareLinearSolve(const TimeContext&     ctx,
                                  VariableBlock          wrt,
                                  linalg::MatrixBuilder& J,
                                  Vector<Real>&          rhs) const
  {
    (void) ctx;
    (void) wrt;
    (void) J;
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
