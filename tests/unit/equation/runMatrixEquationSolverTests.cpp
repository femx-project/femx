#include <iostream>

#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/eq/MatrixNewtonStateSolver.hpp>
#include <femx/inverse/AdjointReducedFunctional.hpp>
#include <femx/inverse/DerivativeCheck.hpp>
#include <femx/inverse/MatrixEquationAdjointSolver.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/DenseLinearSolver.hpp>
#include <femx/system/native/DenseSystemMatrix.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

void resize(Vector& out, index_type size)
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

class LinearAssembledResidualEquation final
  : public equation::AssembledResidualEquation
{
public:
  index_type numStates() const override
  {
    return 2;
  }

  index_type numParams() const override
  {
    return 2;
  }

  index_type numResiduals() const override
  {
    return 2;
  }

  void residual(const Vector& state,
                const Vector& params,
                Vector&       out) const override
  {
    resize(out, numResiduals());
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * params[0] - 2.0 * params[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * params[0] + 4.0 * params[1];
  }

  void assembleStateJac(const Vector&         state,
                        const Vector&         params,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) params;
    out.resize(numResiduals(), numStates());
    out.setZero();
    out.set(0, 0, 2.0);
    out.set(0, 1, 3.0);
    out.set(1, 0, 7.0);
    out.set(1, 1, 11.0);
  }

  void assembleParamJac(const Vector&         state,
                        const Vector&         params,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) params;
    out.resize(numResiduals(), numParams());
    out.setZero();
    out.set(0, 0, 5.0);
    out.set(0, 1, -2.0);
    out.set(1, 0, 13.0);
    out.set(1, 1, 4.0);
  }
};

class TrackingObjective final : public inverse::ObjectiveFunctional
{
public:
  index_type numStates() const override
  {
    return 2;
  }

  index_type numParams() const override
  {
    return 2;
  }

  real_type value(const Vector& state,
                  const Vector& params) const override
  {
    const real_type e0 = state[0] - target_[0];
    const real_type e1 = state[1] - target_[1];
    return 0.5 * (e0 * e0 + e1 * e1)
           + 0.5 * regularization_
                 * (params[0] * params[0] + params[1] * params[1]);
  }

  void stateGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    (void) params;
    resize(out, numStates());
    out[0] = state[0] - target_[0];
    out[1] = state[1] - target_[1];
  }

  void paramGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    (void) state;
    resize(out, numParams());
    out[0] = regularization_ * params[0];
    out[1] = regularization_ * params[1];
  }

private:
  real_type target_[2] = {0.25, -0.75};
  real_type regularization_{0.25};
};

class MatrixEquationSolverTests : public TestBase
{
public:
  TestOutcome defaultApplyUsesSystemJacobians()
  {
    TestStatus status;
    status = true;

    LinearAssembledResidualEquation residual_equation;

    Vector state(2);
    state[0] = -1.48;
    state[1] = 0.89;

    Vector params(2);
    params[0] = 0.05;
    params[1] = -0.02;

    Vector dir(2);
    dir[0] = -0.7;
    dir[1] = 0.4;

    Vector out;
    residual_equation.applyStateJac(state, params, dir, out);
    status *= isEqual(out[0], -0.2);
    status *= isEqual(out[1], -0.5);

    residual_equation.applyStateJacT(state, params, dir, out);
    status *= isEqual(out[0], 1.4);
    status *= isEqual(out[1], 2.3);

    residual_equation.applyParamJac(state, params, dir, out);
    status *= isEqual(out[0], -4.3);
    status *= isEqual(out[1], -7.5);

    residual_equation.applyParamJacT(state, params, dir, out);
    status *= isEqual(out[0], 1.7);
    status *= isEqual(out[1], 3.0);

    return status.report(__func__);
  }

  TestOutcome systemStateAndAdjointSolversWork()
  {
    TestStatus status;
    status = true;

    LinearAssembledResidualEquation residual_equation;
    system::DenseSystemMatrix       state_jac;
    system::DenseLinearSolver       lin_solver;

    equation::MatrixNewtonStateSolver state_solver(
        residual_equation, state_jac, lin_solver);
    inverse::MatrixEquationAdjointSolver adj_solver(
        residual_equation, state_jac, lin_solver);

    Vector params(2);
    params[0] = 0.05;
    params[1] = -0.02;

    Vector state;
    state_solver.solve(params, state);
    status *= isEqual(state[0], -1.48);
    status *= isEqual(state[1], 0.89);

    Vector residual;
    residual_equation.residual(state, params, residual);
    status *= isEqual(residual[0], 0.0);
    status *= isEqual(residual[1], 0.0);

    Vector rhs(2);
    rhs[0] = -1.73;
    rhs[1] = 1.64;

    Vector adjoint;
    adj_solver.solve(state, params, rhs, adjoint);
    status *= isEqual(2.0 * adjoint[0] + 7.0 * adjoint[1], rhs[0]);
    status *= isEqual(3.0 * adjoint[0] + 11.0 * adjoint[1], rhs[1]);

    return status.report(__func__);
  }

  TestOutcome systemReducedGradientChecks()
  {
    TestStatus status;
    status = true;

    LinearAssembledResidualEquation   residual_equation;
    system::DenseSystemMatrix         state_jac;
    system::DenseLinearSolver         lin_solver;
    equation::MatrixNewtonStateSolver state_solver(
        residual_equation, state_jac, lin_solver);
    inverse::MatrixEquationAdjointSolver adj_solver(
        residual_equation, state_jac, lin_solver);
    TrackingObjective objective;

    inverse::AdjointReducedFunctional functional(
        state_solver, adj_solver, residual_equation, objective);

    Vector params(2);
    params[0] = 0.05;
    params[1] = -0.02;

    Vector dir(2);
    dir[0] = -0.7;
    dir[1] = 0.4;

    const inverse::DerivativeCheck check(1.0e-6);
    status *= check.reducedGrad(functional, params, dir)
                  .passed(1.0e-7, 1.0e-7);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running matrix equation solver tests:\n";

  femx::tests::MatrixEquationSolverTests test;

  femx::tests::TestingResults result;
  result += test.defaultApplyUsesSystemJacobians();
  result += test.systemStateAndAdjointSolversWork();
  result += test.systemReducedGradientChecks();

  return result.summary();
}
