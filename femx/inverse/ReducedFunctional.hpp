#pragma once

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/linalg/Backend.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/state/Residual.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx::inverse
{

/** @brief Backend-independent stationary reduced objective and adjoint solve. */
template <class Backend>
class ReducedFunctional final
{
  static_assert(linalg::is_backend_v<Backend>,
                "ReducedFunctional requires a valid backend type");
  static_assert(Backend::space == MemorySpace::Host,
                "ReducedFunctional requires Host objective storage");

public:
  using Vec         = typename Backend::Vec;
  using Mat         = typename Backend::Mat;
  using Ctx         = typename Backend::Ctx;
  using Res         = state::Residual<Backend>;
  using StateSolver = state::StateSolver<Backend>;
  using LinSolver   = linalg::LinearSolver<Backend>;

  ReducedFunctional(StateSolver&     state_solver,
                    Mat&             adj_jac,
                    LinSolver&       adj_solver,
                    const Objective& obj)
    : state_solver_(state_solver),
      res_(state_solver.residual()),
      adj_jac_(adj_jac),
      adj_solver_(adj_solver),
      ctx_(state_solver.context()),
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
    obj_.stateGrad(state_, prm, state_grad_);
    checkSize(state_grad_, dims_.num_states);

    res_.assembleStateJac(state_, prm, adj_jac_, ctx_);
    finalize(adj_jac_, ctx_);
    adj_solver_.solveT(adj_jac_, state_grad_, adj_, ctx_);
    checkSize(adj_, dims_.num_res);

    obj_.paramGrad(state_, prm, prm_grad_);
    res_.applyParamJacT(
        state_, prm, adj_, res_prm_adj_, ctx_);
    checkSize(prm_grad_, numParams());
    checkSize(res_prm_adj_, numParams());
    axpby(-1.0,
          res_prm_adj_.view(),
          1.0,
          prm_grad_.view(),
          ctx_);
    copy(prm_grad_.view(), out, ctx_);
    ctx_.synchronize();
  }

  static void checkSize(const Vec& vec, Index expected)
  {
    require(vec.size() == expected,
            "ReducedFunctional vector size mismatch");
  }

  StateSolver&      state_solver_;
  const Res&        res_;
  Mat&              adj_jac_;
  LinSolver&        adj_solver_;
  Ctx&              ctx_;
  const Objective&  obj_;
  state::Dimensions dims_;
  Vec               state_;
  Vec               state_grad_;
  Vec               adj_;
  Vec               prm_grad_;
  Vec               res_prm_adj_;
};

using HostReducedFunctional =
    ReducedFunctional<linalg::HostCsrBackend>;

} // namespace femx::inverse
