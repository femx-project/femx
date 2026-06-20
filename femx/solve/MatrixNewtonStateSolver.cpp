#include <stdexcept>

#include <femx/solve/MatrixNewtonStateSolver.hpp>
#include <femx/problem/Linearization.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/solve/Newton.hpp>

using namespace femx::algebra;

namespace
{

class StateMatrixLinearization final : public femx::problem::Linearization
{
public:
  explicit StateMatrixLinearization(femx::algebra::SystemMatrix& state_jac)
    : state_jac_(state_jac)
  {
  }

  femx::algebra::SystemMatrix& stateMatrix()
  {
    return state_jac_;
  }

  const femx::algebra::LinearOperator& stateJacobian() const override
  {
    return state_jac_;
  }

  const femx::algebra::LinearOperator& paramJacobian() const override
  {
    throw std::runtime_error(
        "StateMatrixLinearization does not expose parameter Jacobian");
  }

private:
  femx::algebra::SystemMatrix& state_jac_;
};

class MatrixNewtonResidualAdapter final : public femx::problem::Residual
{
public:
  explicit MatrixNewtonResidualAdapter(
      const femx::problem::MatrixResidualEquation& eq)
    : eq_(eq)
  {
  }

  femx::problem::Dimensions dimensions() const override
  {
    return {eq_.numStates(), eq_.numParams(), eq_.numRes()};
  }

  void residual(const femx::Vector<femx::Real>& state,
                const femx::Vector<femx::Real>& prm,
                femx::Vector<femx::Real>&       out) const override
  {
    eq_.res(state, prm, out);
  }

  void linearize(const femx::Vector<femx::Real>& state,
                 const femx::Vector<femx::Real>& prm,
                 femx::problem::Linearization&   out) const override
  {
    auto* matrix_out = dynamic_cast<StateMatrixLinearization*>(&out);
    if (matrix_out == nullptr)
    {
      throw std::runtime_error(
          "MatrixNewtonResidualAdapter requires StateMatrixLinearization");
    }
    eq_.assembleStateJac(state, prm, matrix_out->stateMatrix());
    matrix_out->stateMatrix().finalize();
  }

private:
  const femx::problem::MatrixResidualEquation& eq_;
};

} // namespace

namespace femx
{
namespace solve
{

MatrixNewtonStateSolver::MatrixNewtonStateSolver(
    const MatrixResidualEquation& eq,
    SystemMatrix&                 state_jac,
    LinearSolver&                 lin_solver)
  : eq_(eq),
    state_jac_(state_jac),
    lin_solver_(lin_solver)
{
  if (eq_.numRes() != eq_.numStates())
  {
    throw std::runtime_error(
        "MatrixNewtonStateSolver requires square state residual dimensions");
  }
}

MatrixNewtonStateSolverOptions& MatrixNewtonStateSolver::options()
{
  return options_;
}

const MatrixNewtonStateSolverOptions& MatrixNewtonStateSolver::options() const
{
  return options_;
}

void MatrixNewtonStateSolver::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error(
        "MatrixNewtonStateSolver initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void MatrixNewtonStateSolver::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

Index MatrixNewtonStateSolver::numStates() const
{
  return eq_.numStates();
}

Index MatrixNewtonStateSolver::numParams() const
{
  return eq_.numParams();
}

void MatrixNewtonStateSolver::solve(const Vector<Real>& prm,
                                    Vector<Real>&       state)
{
  MatrixNewtonResidualAdapter adapter(eq_);
  StateMatrixLinearization    linearization(state_jac_);
  solve::Newton               newton(adapter, linearization, lin_solver_);
  newton.options().max_its            = options_.max_its;
  newton.options().residual_tolerance = options_.res_tol;
  newton.options().step_tolerance     = options_.step_tolerance;
  if (has_init_state_)
  {
    newton.setInitialState(init_state_);
  }
  newton.solve(prm, state);
}

void MatrixNewtonStateSolver::initializeState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resize(state, numStates());
}

void MatrixNewtonStateSolver::resize(Vector<Real>& out,
                                     Index         size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

} // namespace solve
} // namespace femx
