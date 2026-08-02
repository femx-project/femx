#pragma once

#include <stdexcept>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/LinearSystem.hpp>
#include <femx/state/Residual.hpp>

namespace femx::state
{

/**
 * @brief Define a stationary state solver in one memory space.
 *
 * Finds a state `x` that satisfies the discrete state equation
 * `R(x, p) = 0` at parameters `p`.
 */
template <MemorySpace Space>
class StateSolver
{
public:
  using Vec = Vector<Space, Real>;
  using Ctx = linalg::Context<Space>;
  using Res = Residual<Space>;

  virtual ~StateSolver() = default;

  virtual Index numStates() const noexcept = 0;
  virtual Index numParams() const noexcept = 0;
  virtual Index numRes() const noexcept    = 0;

  virtual const Res& residual() const noexcept = 0;
  virtual Ctx&       context() const noexcept  = 0;

  /**
   * @brief Solve for a state at the supplied parameter point.
   *
   * The default empty parameter vector is valid only when `numParams()` is zero.
   */
  virtual void solve(Vec& state, const Vec& prm = Vec{}) = 0;
};

/**
 * @brief Solve a stationary state equation of the form `A(p) x = b(p)`.
 *
 * Written as a residual, this system is `R(x, p) = A(p) x - b(p) = 0`.
 * This solver assembles `R(0, p) = -b(p)` and `J = dR/dx = A(p)`, then
 * asks the supplied linear system to solve `J x = -R(0, p)`, which
 * corresponds to the original linear system `A(p) x = b(p)`.
 */
template <MemorySpace Space>
class LinearStateSolver final : public StateSolver<Space>
{
public:
  using Vec    = Vector<Space, Real>;
  using Ctx    = linalg::Context<Space>;
  using Res    = Residual<Space>;
  using System = linalg::LinearSystem<Space>;

  LinearStateSolver(const Res& res, System& system)
    : res_(res),
      system_(system),
      ctx_(system.context()),
      dims_(res.dims()),
      zero_(dims_.num_states),
      res_vec_(dims_.num_res),
      rhs_(dims_.num_res)
  {
    require(dims_.num_res == dims_.num_states,
            "LinearStateSolver requires square residual dimensions");
  }

  Index numStates() const noexcept override
  {
    return dims_.num_states;
  }

  Index numParams() const noexcept override
  {
    return dims_.num_param;
  }

  Index numRes() const noexcept override
  {
    return dims_.num_res;
  }

  const Res& residual() const noexcept override
  {
    return res_;
  }

  Ctx& context() const noexcept override
  {
    return ctx_;
  }

  void solve(Vec& state, const Vec& prm = Vec{}) override
  {
    auto& vec_handler = ctx_.vectorHandler();
    require(prm.size() == numParams(),
            "LinearStateSolver parameter size mismatch");

    auto& jac = system_.matrix();
    jac.setup(res_.hostPattern());

    // Assemble res_vec_ = R(0, prm) = -b(prm).
    res_.assembleResidual(zero_, prm, res_vec_, ctx_);
    require(res_vec_.size() == numRes(),
            "LinearStateSolver residual size mismatch");

    // Set rhs_ = -res_vec_ = b(prm).
    vec_handler.axpby(-1.0, res_vec_.view(), 0.0, rhs_.view());

    // Assemble jac = dR/dx = A(prm)
    res_.assembleJacobian(zero_, prm, jac, ctx_);
    jac.finalize();

    // Solve jac * state = rhs_, which corresponds to A(prm) x = b(prm).
    system_.solve(rhs_.view(), state);
    ctx_.sync();

    require(state.size() == numStates(),
            "LinearStateSolver solution size mismatch");
  }

private:
  const Res& res_;
  System&    system_;
  Ctx&       ctx_;
  Dimensions dims_;
  Vec        zero_;
  Vec        res_vec_;
  Vec        rhs_;
};

struct NewtonStateOptions
{
  Index max_its{20};
  Real  res_tol{1.0e-10};
  Real  step_tol{0.0};
};

/**
 * @brief Solve a Host stationary residual with Newton's method.
 */
template <MemorySpace Space>
class NewtonStateSolver final : public StateSolver<Space>
{
  static_assert(Space == MemorySpace::Host,
                "NewtonStateSolver requires Host state storage");

public:
  using Vec    = Vector<Space, Real>;
  using Ctx    = linalg::Context<Space>;
  using Res    = Residual<Space>;
  using System = linalg::LinearSystem<Space>;

  NewtonStateSolver(const Res& res, System& system)
    : res_(res),
      system_(system),
      ctx_(system.context()),
      dims_(res.dims()),
      init_(dims_.num_states),
      res_vec_(dims_.num_res),
      rhs_(dims_.num_res),
      step_(dims_.num_states)
  {
    require(dims_.num_res == dims_.num_states,
            "NewtonStateSolver requires square residual dimensions");
  }

  NewtonStateOptions& opts() noexcept
  {
    return opts_;
  }

  const NewtonStateOptions& opts() const noexcept
  {
    return opts_;
  }

  void setInitialState(const Vec& state)
  {
    require(state.size() == numStates(),
            "NewtonStateSolver initial state size mismatch");

    auto& vec_handler = ctx_.vectorHandler();
    vec_handler.copy(state.view(), init_);

    ctx_.sync();
    has_init_ = true;
  }

  void clearInitialState() noexcept
  {
    has_init_ = false;
  }

  Index numStates() const noexcept override
  {
    return dims_.num_states;
  }

  Index numParams() const noexcept override
  {
    return dims_.num_param;
  }

  Index numRes() const noexcept override
  {
    return dims_.num_res;
  }

  const Res& residual() const noexcept override
  {
    return res_;
  }

  Ctx& context() const noexcept override
  {
    return ctx_;
  }

  void solve(Vec& state, const Vec& prm = Vec{}) override
  {
    auto& vec_handler = ctx_.vectorHandler();

    require(prm.size() == numParams(),
            "NewtonStateSolver parameter size mismatch");
    initState(state);

    auto& jac = system_.matrix();
    jac.setup(res_.hostPattern());

    for (Index i = 0; i <= opts_.max_its; ++i)
    {
      res_.assembleResidual(state, prm, res_vec_, ctx_);
      require(res_vec_.size() == numRes(),
              "NewtonStateSolver residual size mismatch");

      if (vec_handler.squaredNorm(res_vec_.view())
          <= opts_.res_tol * opts_.res_tol)
      {
        return;
      }
      if (i == opts_.max_its)
      {
        break;
      }

      vec_handler.axpby(-1.0, res_vec_.view(), 0.0, rhs_.view());

      if (i > 0)
      {
        jac.setup(res_.hostPattern());
      }

      res_.assembleJacobian(state, prm, jac, ctx_);
      jac.finalize();

      system_.solve(rhs_.view(), step_);
      require(step_.size() == numStates(),
              "NewtonStateSolver step size mismatch");

      vec_handler.axpby(1.0, step_.view(), 1.0, state.view());
      ctx_.sync();

      if (vec_handler.squaredNorm(step_.view())
          <= opts_.step_tol * opts_.step_tol)
      {
        return;
      }
    }
    throw std::runtime_error("NewtonStateSolver failed to converge");
  }

private:
  void initState(Vec& state)
  {
    auto& vec_handler = ctx_.vectorHandler();
    if (state.size() != numStates())
    {
      state.resize(numStates());
    }
    if (has_init_)
    {
      vec_handler.copy(init_.view(), state);
    }
    else
    {
      vec_handler.zero(state.view());
    }
  }

  const Res&         res_;
  System&            system_;
  Ctx&               ctx_;
  Dimensions         dims_;
  NewtonStateOptions opts_;
  Vec                init_;
  Vec                res_vec_;
  Vec                rhs_;
  Vec                step_;
  bool               has_init_{false};
};

using HostStateSolver         = StateSolver<MemorySpace::Host>;
using HostLinearStateSolver   = LinearStateSolver<MemorySpace::Host>;
using DeviceLinearStateSolver = LinearStateSolver<MemorySpace::Device>;
using HostNewtonStateSolver   = NewtonStateSolver<MemorySpace::Host>;

} // namespace femx::state
