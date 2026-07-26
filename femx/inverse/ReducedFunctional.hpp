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

/** @brief Evaluate a stationary reduced objective and its adjoint gradient. */
template <MemorySpace Space>
class ReducedFunctional final
{
  static_assert(Space == MemorySpace::Host,
                "ReducedFunctional requires Host objective storage");

public:
  using Vec         = Vector<Space, Real>;
  using Ctx         = linalg::Context<Space>;
  using Res         = state::Residual<Space>;
  using StateSolver = state::StateSolver<Space>;
  using System      = linalg::LinearSystem<Space>;

  ReducedFunctional(StateSolver&     state_solver,
                    System&          adj_system,
                    const Objective& obj)
    : state_solver_(state_solver),
      res_(state_solver.residual()),
      adj_system_(adj_system),
      ctx_(adj_system.context()),
      obj_(obj),
      dims_(res_.dims()),
      state_(dims_.num_states),
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

  Index numParams() const noexcept
  {
    return dims_.num_param;
  }

  Real value(const Vec& prm)
  {
    checkPrm(prm);
    state_solver_.solve(prm, state_);
    return obj_.value(state_, prm);
  }

  void grad(const Vec& prm, Vec& out)
  {
    checkPrm(prm);
    state_solver_.solve(prm, state_);
    gradAt(prm, out);
  }

  Real valueGrad(const Vec& prm, Vec& out)
  {
    checkPrm(prm);
    state_solver_.solve(prm, state_);
    const Real val = obj_.value(state_, prm);
    gradAt(prm, out);
    return val;
  }

private:
  void checkPrm(const Vec& prm) const
  {
    require(prm.size() == numParams(),
            "ReducedFunctional parameter size mismatch");
  }

  void gradAt(const Vec& prm, Vec& out)
  {
    auto& vec_handler = ctx_.vectors();
    obj_.stateGrad(state_, prm, state_grad_);
    checkSize(state_grad_, dims_.num_states);

    auto& jac = adj_system_.jacobian();
    jac.begin(res_.hostPattern());
    res_.assembleStateJac(state_, prm, jac, ctx_);
    jac.finalize();
    adj_system_.solveT(state_grad_.view(), adj_);
    checkSize(adj_, dims_.num_res);

    obj_.paramGrad(state_, prm, prm_grad_);
    res_.applyParamJacT(
        state_, prm, adj_, res_prm_adj_, ctx_);
    checkSize(prm_grad_, numParams());
    checkSize(res_prm_adj_, numParams());
    vec_handler.axpby(-1.0,
                      res_prm_adj_.view(),
                      1.0,
                      prm_grad_.view());
    vec_handler.copy(prm_grad_.view(), out);
    ctx_.sync();
  }

  static void checkSize(const Vec& vec, Index expected)
  {
    require(vec.size() == expected,
            "ReducedFunctional vector size mismatch");
  }

  StateSolver&      state_solver_;
  const Res&        res_;
  System&           adj_system_;
  Ctx&              ctx_;
  const Objective&  obj_;
  state::Dimensions dims_;
  Vec               state_;
  Vec               state_grad_;
  Vec               adj_;
  Vec               prm_grad_;
  Vec               res_prm_adj_;
};

using HostReducedFunctional = ReducedFunctional<MemorySpace::Host>;

} // namespace femx::inverse
