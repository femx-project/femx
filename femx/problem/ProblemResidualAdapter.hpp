#pragma once

#include <stdexcept>

#include <femx/problem/MatrixResidualEquation.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace problem
{

class ResidualEquationLinearization final : public problem::Linearization
{
public:
  ResidualEquationLinearization()
    : state_op_(*this),
      param_op_(*this)
  {
  }

  void reset(const ResidualEquation& eq,
             const Vector<Real>&     state,
             const Vector<Real>&     prm)
  {
    eq_    = &eq;
    state_ = &state;
    prm_   = &prm;
  }

  const algebra::LinearOperator& stateJacobian() const override
  {
    return state_op_;
  }

  const algebra::LinearOperator& paramJacobian() const override
  {
    return param_op_;
  }

private:
  class StateOperator final : public algebra::LinearOperator
  {
  public:
    explicit StateOperator(const ResidualEquationLinearization& owner)
      : owner_(owner)
    {
    }

    Index numRows() const override
    {
      return owner_.equation().numRes();
    }

    Index numCols() const override
    {
      return owner_.equation().numStates();
    }

    void apply(const Vector<Real>& dir, Vector<Real>& out) const override
    {
      owner_.equation().applyStateJac(
          owner_.state(), owner_.params(), dir, out);
    }

    void applyT(const Vector<Real>& dir, Vector<Real>& out) const override
    {
      owner_.equation().applyStateJacT(
          owner_.state(), owner_.params(), dir, out);
    }

  private:
    const ResidualEquationLinearization& owner_;
  };

  class ParamOperator final : public algebra::LinearOperator
  {
  public:
    explicit ParamOperator(const ResidualEquationLinearization& owner)
      : owner_(owner)
    {
    }

    Index numRows() const override
    {
      return owner_.equation().numRes();
    }

    Index numCols() const override
    {
      return owner_.equation().numParams();
    }

    void apply(const Vector<Real>& dir, Vector<Real>& out) const override
    {
      owner_.equation().applyParamJac(
          owner_.state(), owner_.params(), dir, out);
    }

    void applyT(const Vector<Real>& dir, Vector<Real>& out) const override
    {
      owner_.equation().applyParamJacT(
          owner_.state(), owner_.params(), dir, out);
    }

  private:
    const ResidualEquationLinearization& owner_;
  };

  const ResidualEquation& equation() const
  {
    if (eq_ == nullptr)
    {
      throw std::runtime_error(
          "ResidualEquationLinearization is not initialized");
    }
    return *eq_;
  }

  const Vector<Real>& state() const
  {
    if (state_ == nullptr)
    {
      throw std::runtime_error(
          "ResidualEquationLinearization state is not initialized");
    }
    return *state_;
  }

  const Vector<Real>& params() const
  {
    if (prm_ == nullptr)
    {
      throw std::runtime_error(
          "ResidualEquationLinearization parameters are not initialized");
    }
    return *prm_;
  }

private:
  const ResidualEquation* eq_{nullptr};
  const Vector<Real>*     state_{nullptr};
  const Vector<Real>*     prm_{nullptr};
  StateOperator           state_op_;
  ParamOperator           param_op_;
};

class ResidualEquationProblemAdapter final : public problem::Residual
{
public:
  explicit ResidualEquationProblemAdapter(const ResidualEquation& eq)
    : eq_(eq)
  {
  }

  problem::Dimensions dimensions() const override
  {
    return {eq_.numStates(), eq_.numParams(), eq_.numRes()};
  }

  void residual(const Vector<Real>& state,
                const Vector<Real>& prm,
                Vector<Real>&       out) const override
  {
    eq_.res(state, prm, out);
  }

  void linearize(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 problem::Linearization& out) const override
  {
    auto* operator_out = dynamic_cast<ResidualEquationLinearization*>(&out);
    if (operator_out == nullptr)
    {
      throw std::runtime_error(
          "ResidualEquationProblemAdapter requires "
          "ResidualEquationLinearization output");
    }
    operator_out->reset(eq_, state, prm);
  }

private:
  const ResidualEquation& eq_;
};

class MatrixResidualEquationProblemAdapter final : public problem::Residual
{
public:
  explicit MatrixResidualEquationProblemAdapter(
      const MatrixResidualEquation& eq)
    : eq_(eq)
  {
  }

  problem::Dimensions dimensions() const override
  {
    return {eq_.numStates(), eq_.numParams(), eq_.numRes()};
  }

  void residual(const Vector<Real>& state,
                const Vector<Real>& prm,
                Vector<Real>&       out) const override
  {
    eq_.res(state, prm, out);
  }

  void linearize(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 problem::Linearization& out) const override
  {
    auto* matrix_out = dynamic_cast<problem::MatrixLinearization*>(&out);
    if (matrix_out == nullptr)
    {
      throw std::runtime_error(
          "MatrixResidualEquationProblemAdapter requires "
          "problem::MatrixLinearization output");
    }

    auto* state_jac =
        dynamic_cast<algebra::SystemMatrix*>(&matrix_out->stateMatrix());
    auto* param_jac =
        dynamic_cast<algebra::SystemMatrix*>(&matrix_out->paramMatrix());
    if (state_jac == nullptr || param_jac == nullptr)
    {
      throw std::runtime_error(
          "MatrixResidualEquationProblemAdapter requires SystemMatrix-backed "
          "matrix operators");
    }

    eq_.assembleStateJac(state, prm, *state_jac);
    state_jac->finalize();
    eq_.assembleParamJac(state, prm, *param_jac);
    param_jac->finalize();
  }

private:
  const MatrixResidualEquation& eq_;
};

} // namespace problem
} // namespace femx
