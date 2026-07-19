#pragma once

#include <stdexcept>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Backend.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/state/Residual.hpp>

namespace femx::state
{

/** @brief Solver contract for one stationary residual backend. */
template <class Backend>
class StateSolver
{
  static_assert(linalg::is_backend_v<Backend>,
                "StateSolver requires a valid backend type");

public:
  using Vec = typename Backend::Vec;
  using Ctx = typename Backend::Ctx;
  using Res = Residual<Backend>;

  virtual ~StateSolver() = default;

  virtual Index numStates() const noexcept = 0;
  virtual Index numParams() const noexcept = 0;
  virtual Index numRes() const noexcept    = 0;

  virtual const Res& residual() const noexcept = 0;
  virtual Ctx&       context() const noexcept  = 0;

  virtual void solve(const Vec& prm, Vec& state) = 0;
};

/** @brief State solver for affine-linear stationary residuals. */
template <class Backend>
class LinearStateSolver final : public StateSolver<Backend>
{
public:
  using Vec    = typename Backend::Vec;
  using Mat    = typename Backend::Mat;
  using Ctx    = typename Backend::Ctx;
  using Res    = Residual<Backend>;
  using Solver = linalg::LinearSolver<Backend>;

  LinearStateSolver(const Res& res, Mat& jac, Solver& solver, Ctx& ctx)
    : res_(res),
      jac_(jac),
      solver_(solver),
      ctx_(ctx),
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

  void solve(const Vec& prm, Vec& state) override
  {
    require(prm.size() == numParams(),
            "LinearStateSolver parameter size mismatch");

    res_.res(zero_, prm, res_vec_, ctx_);
    require(res_vec_.size() == numRes(),
            "LinearStateSolver residual size mismatch");
    axpby(-1.0, res_vec_.view(), 0.0, rhs_.view(), ctx_);
    res_.assembleStateJac(zero_, prm, jac_, ctx_);
    finalize(jac_, ctx_);
    solver_.solve(jac_, rhs_, state, ctx_);
    ctx_.synchronize();
    require(state.size() == numStates(),
            "LinearStateSolver solution size mismatch");
  }

private:
  const Res& res_;
  Mat&       jac_;
  Solver&    solver_;
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

/** @brief Newton state solver for Host-storage backends. */
template <class Backend>
class NewtonStateSolver final : public StateSolver<Backend>
{
  static_assert(Backend::space == MemorySpace::Host,
                "NewtonStateSolver requires Host state storage");

public:
  using Vec    = typename Backend::Vec;
  using Mat    = typename Backend::Mat;
  using Ctx    = typename Backend::Ctx;
  using Res    = Residual<Backend>;
  using Solver = linalg::LinearSolver<Backend>;

  NewtonStateSolver(const Res& res, Mat& jac, Solver& solver, Ctx& ctx)
    : res_(res),
      jac_(jac),
      solver_(solver),
      ctx_(ctx),
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
    copy(state.view(), init_, ctx_);
    ctx_.synchronize();
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

  void solve(const Vec& prm, Vec& state) override
  {
    require(prm.size() == numParams(),
            "NewtonStateSolver parameter size mismatch");
    initState(state);

    for (Index i = 0; i <= opts_.max_its; ++i)
    {
      res_.res(state, prm, res_vec_, ctx_);
      require(res_vec_.size() == numRes(),
              "NewtonStateSolver residual size mismatch");
      if (squaredNorm(res_vec_.view(), ctx_)
          <= opts_.res_tol * opts_.res_tol)
      {
        return;
      }
      if (i == opts_.max_its)
      {
        break;
      }

      axpby(-1.0, res_vec_.view(), 0.0, rhs_.view(), ctx_);
      res_.assembleStateJac(state, prm, jac_, ctx_);
      finalize(jac_, ctx_);
      solver_.solve(jac_, rhs_, step_, ctx_);
      require(step_.size() == numStates(),
              "NewtonStateSolver step size mismatch");
      axpby(1.0, step_.view(), 1.0, state.view(), ctx_);
      ctx_.synchronize();

      if (squaredNorm(step_.view(), ctx_)
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
    if (state.size() != numStates())
    {
      state.resize(numStates());
    }
    if (has_init_)
    {
      copy(init_.view(), state, ctx_);
    }
    else
    {
      axpby(0.0, state.view(), 0.0, state.view(), ctx_);
    }
  }

  const Res&         res_;
  Mat&               jac_;
  Solver&            solver_;
  Ctx&               ctx_;
  Dimensions         dims_;
  NewtonStateOptions opts_;
  Vec                init_;
  Vec                res_vec_;
  Vec                rhs_;
  Vec                step_;
  bool               has_init_{false};
};

using HostStateSolver       = StateSolver<linalg::HostCsrBackend>;
using HostLinearStateSolver = LinearStateSolver<linalg::HostCsrBackend>;
using HostNewtonStateSolver = NewtonStateSolver<linalg::HostCsrBackend>;

} // namespace femx::state
