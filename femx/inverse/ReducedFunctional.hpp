#pragma once

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/LinearSystem.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/Residual.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx::inverse
{

/**
 * @brief Evaluate a stationary reduced objective and its adjoint gradient.
 *
 * The objective and TAO-facing parameter vectors remain on Host. State,
 * residual, Jacobian, and adjoint operations use `Space`.
 */
template <MemorySpace Space>
class ReducedFunctional final
{
public:
  static constexpr MemorySpace space = Space; ///< Linear algebra memory space.

  using Vec         = Vector<space, Real>;
  using Ctx         = linalg::Context<space>;
  using Res         = state::Residual<Space>;
  using StateSolver = state::StateSolver<Space>;
  using System      = linalg::LinearSystem<Space>;

  /**
   * @brief Bind forward and adjoint systems to a Host objective.
   *
   * @param[in,out] state_solver - Forward state solver.
   * @param[in,out] adj_system - Linear system used for adjoint solves.
   * @param[in] obj - Host objective with matching dimensions.
   */
  ReducedFunctional(StateSolver&     state_solver,
                    System&          adj_system,
                    const Objective& obj)
    : state_solver_(state_solver),
      res_(state_solver.residual()),
      adj_system_(adj_system),
      fwd_ctx_(state_solver.context()),
      adj_ctx_(adj_system.context()),
      obj_(obj),
      dims_(res_.dims()),
      state_(dims_.num_states),
      prm_(dims_.num_param),
      state_grad_(dims_.num_states),
      adj_(dims_.num_res),
      prm_grad_(dims_.num_param),
      res_prm_adj_(dims_.num_param)
  {
    require(dims_.num_states == state_solver_.numStates()
                && dims_.num_param == state_solver_.numParams()
                && dims_.num_res == state_solver_.numRes()
                && dims_.num_states == obj_.numStates()
                && dims_.num_param == obj_.numParams()
                && dims_.num_res == dims_.num_states,
            "ReducedFunctional received inconsistent dimensions");
  }

  ReducedFunctional(const ReducedFunctional&)            = delete;
  ReducedFunctional& operator=(const ReducedFunctional&) = delete;
  ReducedFunctional(ReducedFunctional&&)                 = delete;
  ReducedFunctional& operator=(ReducedFunctional&&)      = delete;

  /** @brief Return the number of optimization parameters. */
  Index numParams() const noexcept
  {
    return dims_.num_param;
  }

  /**
   * @brief Evaluate the reduced objective.
   *
   * @param[in] prm - Host optimization parameters.
   * @return Reduced-objective value.
   */
  Real value(const HostVector<Real>& prm)
  {
    solveState(prm);
    return obj_.value(h_state_, h_prm_);
  }

  /**
   * @brief Evaluate the reduced gradient.
   *
   * @param[in] prm - Host optimization parameters.
   * @param[out] out - Host reduced gradient.
   */
  void grad(const HostVector<Real>& prm,
            HostVector<Real>&       out)
  {
    solveState(prm);
    gradAtState(out);
  }

  /**
   * @brief Evaluate the reduced objective and gradient together.
   *
   * @param[in] prm - Host optimization parameters.
   * @param[out] out - Host reduced gradient.
   * @return Reduced-objective value.
   */
  Real valueGrad(const HostVector<Real>& prm,
                 HostVector<Real>&       out)
  {
    solveState(prm);
    const Real val = obj_.value(h_state_, h_prm_);
    gradAtState(out);
    return val;
  }

private:
  void solveState(const HostVector<Real>& prm)
  {
    require(prm.size() == numParams(),
            "ReducedFunctional parameter size mismatch");
    h_prm_ = prm;

    auto& vec_handler = fwd_ctx_.vectors();
    vec_handler.copy(h_prm_.view(), prm_);
    state_solver_.solve(state_, prm_);
    vec_handler.copy(state_.view(), h_state_);
    fwd_ctx_.sync();
    checkSize(state_, dims_.num_states);
  }

  void gradAtState(HostVector<Real>& out)
  {
    auto& vec_handler = adj_ctx_.vectors();
    obj_.stateGrad(
        h_state_, h_prm_, h_state_grad_);
    require(h_state_grad_.size() == dims_.num_states,
            "ReducedFunctional objective state-gradient size mismatch");
    vec_handler.copy(h_state_grad_.view(), state_grad_);

    auto& jac = adj_system_.jacobian();
    jac.setup(res_.hostPattern());
    res_.assembleJacobian(state_, prm_, jac, adj_ctx_);
    jac.finalize();
    adj_system_.solveT(state_grad_.view(), adj_);
    checkSize(adj_, dims_.num_res);

    obj_.paramGrad(h_state_, h_prm_, h_prm_grad_);
    require(h_prm_grad_.size() == numParams(),
            "ReducedFunctional objective parameter-gradient size mismatch");
    vec_handler.copy(h_prm_grad_.view(), prm_grad_);

    res_.applyParamJacT(
        state_, prm_, adj_, res_prm_adj_, adj_ctx_);
    checkSize(res_prm_adj_, numParams());
    vec_handler.axpby(
        -1.0, res_prm_adj_.view(), 1.0, prm_grad_.view());
    vec_handler.copy(prm_grad_.view(), out);
    adj_ctx_.sync();
  }

  static void checkSize(const Vec& vec, Index expected)
  {
    require(vec.size() == expected,
            "ReducedFunctional vector size mismatch");
  }

  StateSolver&      state_solver_; ///< Forward state solver.
  const Res&        res_;          ///< State residual.
  System&           adj_system_;   ///< Adjoint linear system.
  Ctx&              fwd_ctx_;      ///< Forward linear algebra context.
  Ctx&              adj_ctx_;      ///< Adjoint linear algebra context.
  const Objective&  obj_;          ///< Host objective.
  state::Dimensions dims_;         ///< Residual dimensions.
  HostVector<Real>  h_state_;      ///< Host state.
  HostVector<Real>  h_prm_;        ///< Host parameters.
  HostVector<Real>  h_state_grad_; ///< Host state gradient.
  HostVector<Real>  h_prm_grad_;   ///< Host parameter gradient.
  Vec               state_;        ///< State in `Space`.
  Vec               prm_;          ///< Parameters in `Space`.
  Vec               state_grad_;   ///< State gradient in `Space`.
  Vec               adj_;          ///< Residual adjoint in `Space`.
  Vec               prm_grad_;     ///< Parameter gradient in `Space`.
  Vec               res_prm_adj_;  ///< Residual parameter-adjoint product.
};

using HostReducedFunctional = ReducedFunctional<MemorySpace::Host>;
using DeviceReducedFunctional =
    ReducedFunctional<MemorySpace::Device>;

} // namespace femx::inverse
