#pragma once

#include <stdexcept>

#include <femx/algebra/MatrixBuilder.hpp>
#include <femx/problem/TimeMatrixResidualEquation.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace problem
{

class TimeResidualEquationProblemAdapter final
    : public problem::TimeResidual
{
public:
  explicit TimeResidualEquationProblemAdapter(
      const TimeResidualEquation& eq)
    : eq_(eq)
  {
  }

  problem::TimeDimensions dimensions() const override
  {
    return {eq_.numSteps(), eq_.numStates(), eq_.numParams(), eq_.numRes()};
  }

  void residual(const problem::TimeContext& ctx,
                Vector<Real>&               out) const override
  {
    eq_.res(ctx.step, nextState(ctx), previousState(ctx), params(ctx), out);
  }

  void applyJacobian(const problem::TimeContext& ctx,
                     problem::VariableBlock     wrt,
                     const Vector<Real>&        dir,
                     Vector<Real>&              out) const override
  {
    switch (wrt)
    {
    case problem::VariableBlock::PreviousState:
      eq_.applyPreviousStateJac(
          ctx.step, nextState(ctx), previousState(ctx), params(ctx), dir, out);
      return;

    case problem::VariableBlock::NextState:
      eq_.applyNextStateJac(
          ctx.step, nextState(ctx), previousState(ctx), params(ctx), dir, out);
      return;

    case problem::VariableBlock::Parameter:
      eq_.applyParamJac(
          ctx.step, nextState(ctx), previousState(ctx), params(ctx), dir, out);
      return;
    }

    throw std::runtime_error("Unsupported time residual variable block");
  }

  void applyJacobianT(const problem::TimeContext& ctx,
                      problem::VariableBlock     wrt,
                      const Vector<Real>&        adjoint,
                      Vector<Real>&              out) const override
  {
    switch (wrt)
    {
    case problem::VariableBlock::PreviousState:
      eq_.applyPreviousStateJacT(ctx.step,
                                 nextState(ctx),
                                 previousState(ctx),
                                 params(ctx),
                                 adjoint,
                                 out);
      return;

    case problem::VariableBlock::NextState:
      eq_.applyNextStateJacT(ctx.step,
                             nextState(ctx),
                             previousState(ctx),
                             params(ctx),
                             adjoint,
                             out);
      return;

    case problem::VariableBlock::Parameter:
      eq_.applyParamJacT(ctx.step,
                         nextState(ctx),
                         previousState(ctx),
                         params(ctx),
                         adjoint,
                         out);
      return;
    }

    throw std::runtime_error("Unsupported time residual variable block");
  }

private:
  static const Vector<Real>& previousState(const problem::TimeContext& ctx)
  {
    if (ctx.previous_state == nullptr)
    {
      throw std::runtime_error(
          "Time residual context has no previous state");
    }
    return *ctx.previous_state;
  }

  static const Vector<Real>& nextState(const problem::TimeContext& ctx)
  {
    if (ctx.next_state == nullptr)
    {
      throw std::runtime_error("Time residual context has no next state");
    }
    return *ctx.next_state;
  }

  static const Vector<Real>& params(const problem::TimeContext& ctx)
  {
    if (ctx.prm == nullptr)
    {
      throw std::runtime_error("Time residual context has no parameters");
    }
    return *ctx.prm;
  }

private:
  const TimeResidualEquation& eq_;
};

class TimeMatrixResidualEquationProblemAdapter final
    : public problem::TimeResidual
{
public:
  explicit TimeMatrixResidualEquationProblemAdapter(
      const TimeMatrixResidualEquation& eq)
    : eq_(eq),
      operator_adapter_(eq)
  {
  }

  problem::TimeDimensions dimensions() const override
  {
    return operator_adapter_.dimensions();
  }

  void residual(const problem::TimeContext& ctx,
                Vector<Real>&               out) const override
  {
    operator_adapter_.residual(ctx, out);
  }

  void applyJacobian(const problem::TimeContext& ctx,
                     problem::VariableBlock     wrt,
                     const Vector<Real>&        dir,
                     Vector<Real>&              out) const override
  {
    operator_adapter_.applyJacobian(ctx, wrt, dir, out);
  }

  void applyJacobianT(const problem::TimeContext& ctx,
                      problem::VariableBlock     wrt,
                      const Vector<Real>&        adjoint,
                      Vector<Real>&              out) const override
  {
    operator_adapter_.applyJacobianT(ctx, wrt, adjoint, out);
  }

  bool assembleJacobian(const problem::TimeContext& ctx,
                        problem::VariableBlock     wrt,
                        algebra::MatrixBuilder&    out) const override
  {
    auto* matrix_out = dynamic_cast<algebra::SystemMatrix*>(&out);
    if (matrix_out == nullptr)
    {
      return false;
    }

    switch (wrt)
    {
    case problem::VariableBlock::PreviousState:
      eq_.assemblePrevStateJac(ctx.step,
                               nextState(ctx),
                               previousState(ctx),
                               params(ctx),
                               *matrix_out);
      break;

    case problem::VariableBlock::NextState:
      eq_.assembleNextStateJac(ctx.step,
                               nextState(ctx),
                               previousState(ctx),
                               params(ctx),
                               *matrix_out);
      break;

    case problem::VariableBlock::Parameter:
      eq_.assembleParamJac(ctx.step,
                           nextState(ctx),
                           previousState(ctx),
                           params(ctx),
                           *matrix_out);
      break;
    }

    matrix_out->finalize();
    return true;
  }

private:
  static const Vector<Real>& previousState(const problem::TimeContext& ctx)
  {
    if (ctx.previous_state == nullptr)
    {
      throw std::runtime_error(
          "Time residual context has no previous state");
    }
    return *ctx.previous_state;
  }

  static const Vector<Real>& nextState(const problem::TimeContext& ctx)
  {
    if (ctx.next_state == nullptr)
    {
      throw std::runtime_error("Time residual context has no next state");
    }
    return *ctx.next_state;
  }

  static const Vector<Real>& params(const problem::TimeContext& ctx)
  {
    if (ctx.prm == nullptr)
    {
      throw std::runtime_error("Time residual context has no parameters");
    }
    return *ctx.prm;
  }

private:
  const TimeMatrixResidualEquation&    eq_;
  TimeResidualEquationProblemAdapter   operator_adapter_;
};

} // namespace problem
} // namespace femx
