#pragma once

#include <optional>

#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx
{
class DenseMatrix;

namespace linalg
{
class MatrixOperator;
} // namespace linalg

namespace assembly
{

/** @brief Element-local residual and Jacobian for one time step. */
class HostElementOperator
{
public:
  virtual ~HostElementOperator() = default;

  /** @brief Evaluate one element residual at a time step. */
  virtual void res(Index                  step,
                   Index                  ie,
                   state::TimeHistoryView hist,
                   const HostVector&      nxt,
                   const HostVector&      prm,
                   HostVector&            out) const = 0;

  /** @brief Evaluate one element Jacobian with respect to `wrt`. */
  virtual void jac(Index                  step,
                   Index                  ie,
                   state::VariableBlock   wrt,
                   state::TimeHistoryView hist,
                   const HostVector&      nxt,
                   const HostVector&      prm,
                   DenseMatrix&           out) const = 0;
};

/**
 * @brief state::TimeResidual assembled from elem-local FEM time kernels.
 *
 * HostTimeResidual gathers local state histories, calls element kernels, and
 * scatters residual or Jacobian contributions into global objects.
 */
class HostTimeResidual final : public state::TimeResidual
{
public:
  /** @brief Use one state layout for history and next-state variables. */
  HostTimeResidual(Index                      nstep,
                   fem::DofLayout             res_lyt,
                   fem::DofLayout             state_lyt,
                   const HostElementOperator& op);

  /** @brief Use distinct single-history and next-state layouts. */
  HostTimeResidual(Index                      nstep,
                   fem::DofLayout             res_lyt,
                   fem::DofLayout             hist_lyt,
                   fem::DofLayout             next_lyt,
                   const HostElementOperator& op);

  /** @brief Use distinct layouts for each required history level. */
  HostTimeResidual(Index                      nstep,
                   fem::DofLayout             res_lyt,
                   Array<fem::DofLayout>      hist_lyts,
                   fem::DofLayout             next_lyt,
                   const HostElementOperator& op);

  /** @brief Add an element parameter layout to a single-history residual. */
  HostTimeResidual(Index                      nstep,
                   fem::DofLayout             res_lyt,
                   fem::DofLayout             hist_lyt,
                   fem::DofLayout             next_lyt,
                   fem::DofLayout             prm_lyt,
                   const HostElementOperator& op);

  /** @brief Add an element parameter layout to a multi-history residual. */
  HostTimeResidual(Index                      nstep,
                   fem::DofLayout             res_lyt,
                   Array<fem::DofLayout>      hist_lyts,
                   fem::DofLayout             next_lyt,
                   fem::DofLayout             prm_lyt,
                   const HostElementOperator& op);

  /** @brief Restrict assembly to the half-open element range `[begin, end)`. */
  void setElemRange(Index ie_begin, Index ie_end);

  /** @brief Return global time-residual dimensions. */
  state::TimeDims dims() const override;

  /** @brief Assemble the global residual for one time context. */
  void res(const state::TimeContext& ctx,
           HostVector&               out) const override;

  /** @brief Apply a matrix-free element Jacobian to a direction. */
  void applyJac(const state::TimeContext& ctx,
                state::VariableBlock      wrt,
                const HostVector&         dir,
                HostVector&               out) const override;

  /** @brief Apply the transpose element Jacobian to an adjoint vector. */
  void applyJacT(const state::TimeContext& ctx,
                 state::VariableBlock      wrt,
                 const HostVector&         adj,
                 HostVector&               out) const override;

  /** @brief Assemble one variable-block Jacobian when the target supports it. */
  bool assembleJac(const state::TimeContext& ctx,
                   state::VariableBlock      wrt,
                   linalg::MatrixOperator&   out) const override;

private:
  Index numElems() const;
  Index numPrm() const;
  Index numHistStates() const;

  const fem::DofLayout* layoutFor(state::VariableBlock wrt) const;

  void checkLyts() const;
  void checkCtx(const state::TimeContext& ctx) const;
  void checkDirection(state::VariableBlock wrt,
                      const HostVector&    dir) const;

  void gatherHist(const state::TimeContext& ctx,
                  Index                     ie,
                  HostVector&               hist_e) const;

  void gatherElem(const state::TimeContext& ctx,
                  Index                     ie,
                  HostVector&               hist_e,
                  HostVector&               next_e,
                  HostVector&               prm_e) const;

  state::TimeHistoryView histView(const HostVector& hist_e) const;

  void reduce(HostVector& vec) const;

  static void gather(const fem::DofLayout& lyt,
                     const HostVector&     vec,
                     Index                 ie,
                     HostVector&           vec_e);

  static void gather(const fem::DofLayout& lyt,
                     HostConstVectorView   vec,
                     Index                 ie,
                     HostVector&           vec_e);

  static void gather(const fem::DofLayout& lyt,
                     HostConstVectorView   vec,
                     Index                 ie,
                     HostVectorView        vec_e);

private:
  Index                         nstep_{0};
  fem::DofLayout                res_lyt_;
  Array<fem::DofLayout>         hist_lyts_;
  fem::DofLayout                next_lyt_;
  std::optional<fem::DofLayout> prm_lyt_;
  const HostElementOperator&    op_;
  Index                         ie_begin_{0};
  Index                         ie_end_{0};
};

} // namespace assembly
} // namespace femx
