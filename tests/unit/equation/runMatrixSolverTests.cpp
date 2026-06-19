#include <iostream>

#include <femx/eq/MatrixLinearStateSolver.hpp>
#include <femx/eq/MatrixResidualEquation.hpp>
#include <femx/inverse/AdjointReducedFunctional.hpp>
#include <femx/inverse/DerivativeCheck.hpp>
#include <femx/inverse/MatrixAdjointSolver.hpp>
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

void resize(Vector<Real>& out, Index size)
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

class LinearMatrixResidualEquation final
  : public eq::MatrixResidualEquation
{
public:
  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 2;
  }

  Index numRes() const override
  {
    return 2;
  }

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    resize(out, numRes());
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * prm[0] - 2.0 * prm[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * prm[0] + 4.0 * prm[1];
  }

  void assembleStateJac(const Vector<Real>&   state,
                        const Vector<Real>&   prm,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) prm;
    out.resize(numRes(), numStates());
    out.setZero();
    out.set(0, 0, 2.0);
    out.set(0, 1, 3.0);
    out.set(1, 0, 7.0);
    out.set(1, 1, 11.0);
  }

  void assembleParamJac(const Vector<Real>&   state,
                        const Vector<Real>&   prm,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) prm;
    out.resize(numRes(), numParams());
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
  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 2;
  }

  Real value(const Vector<Real>& state,
             const Vector<Real>& prm) const override
  {
    const Real e0 = state[0] - target_[0];
    const Real e1 = state[1] - target_[1];
    return 0.5 * (e0 * e0 + e1 * e1)
           + 0.5 * regularization_
                 * (prm[0] * prm[0] + prm[1] * prm[1]);
  }

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    (void) prm;
    resize(out, numStates());
    out[0] = state[0] - target_[0];
    out[1] = state[1] - target_[1];
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    (void) state;
    resize(out, numParams());
    out[0] = regularization_ * prm[0];
    out[1] = regularization_ * prm[1];
  }

private:
  Real target_[2] = {0.25, -0.75};
  Real regularization_{0.25};
};

class MatrixSolverTests : public TestBase
{
public:
  TestOutcome defaultApplyUsesSystemJacobians()
  {
    TestStatus status;
    status = true;

    LinearMatrixResidualEquation res_eq;

    Vector<Real> state(2);
    state[0] = -1.48;
    state[1] = 0.89;

    Vector<Real> prm(2);
    prm[0] = 0.05;
    prm[1] = -0.02;

    Vector<Real> dir(2);
    dir[0] = -0.7;
    dir[1] = 0.4;

    Vector<Real> out;
    res_eq.applyStateJac(state, prm, dir, out);
    status *= isEqual(out[0], -0.2);
    status *= isEqual(out[1], -0.5);

    res_eq.applyStateJacT(state, prm, dir, out);
    status *= isEqual(out[0], 1.4);
    status *= isEqual(out[1], 2.3);

    res_eq.applyParamJac(state, prm, dir, out);
    status *= isEqual(out[0], -4.3);
    status *= isEqual(out[1], -7.5);

    res_eq.applyParamJacT(state, prm, dir, out);
    status *= isEqual(out[0], 1.7);
    status *= isEqual(out[1], 3.0);

    return status.report(__func__);
  }

  TestOutcome systemStateAndAdjointSolversWork()
  {
    TestStatus status;
    status = true;

    LinearMatrixResidualEquation res_eq;
    system::DenseSystemMatrix    state_jac;
    system::DenseLinearSolver    lin_solver;

    eq::MatrixLinearStateSolver state_solver(
        res_eq, state_jac, lin_solver);
    inverse::MatrixAdjointSolver adj_solver(
        res_eq, state_jac, lin_solver);

    Vector<Real> prm(2);
    prm[0] = 0.05;
    prm[1] = -0.02;

    Vector<Real> state;
    state_solver.solve(prm, state);
    status *= isEqual(state[0], -1.48);
    status *= isEqual(state[1], 0.89);

    Vector<Real> res;
    res_eq.res(state, prm, res);
    status *= isEqual(res[0], 0.0);
    status *= isEqual(res[1], 0.0);

    Vector<Real> reference(2);
    reference[0] = 3.0;
    reference[1] = -4.0;
    state_solver.setReferenceState(reference);
    state_solver.solve(prm, state);
    status *= isEqual(state[0], -1.48);
    status *= isEqual(state[1], 0.89);
    state_solver.clearReferenceState();

    Vector<Real> rhs(2);
    rhs[0] = -1.73;
    rhs[1] = 1.64;

    Vector<Real> adjoint;
    adj_solver.solve(state, prm, rhs, adjoint);
    status *= isEqual(2.0 * adjoint[0] + 7.0 * adjoint[1], rhs[0]);
    status *= isEqual(3.0 * adjoint[0] + 11.0 * adjoint[1], rhs[1]);

    return status.report(__func__);
  }

  TestOutcome systemReducedGradientChecks()
  {
    TestStatus status;
    status = true;

    LinearMatrixResidualEquation res_eq;
    system::DenseSystemMatrix    state_jac;
    system::DenseLinearSolver    lin_solver;
    eq::MatrixLinearStateSolver  state_solver(
        res_eq, state_jac, lin_solver);
    inverse::MatrixAdjointSolver adj_solver(
        res_eq, state_jac, lin_solver);
    TrackingObjective objective;

    inverse::AdjointReducedFunctional functional(
        state_solver, adj_solver, res_eq, objective);

    Vector<Real> prm(2);
    prm[0] = 0.05;
    prm[1] = -0.02;

    Vector<Real> dir(2);
    dir[0] = -0.7;
    dir[1] = 0.4;

    const inverse::DerivativeCheck check(1.0e-6);
    status *= check.reducedGrad(functional, prm, dir)
                  .passed(1.0e-7, 1.0e-7);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running matrix solver tests:\n";

  femx::tests::MatrixSolverTests test;

  femx::tests::TestingResults result;
  result += test.defaultApplyUsesSystemJacobians();
  result += test.systemStateAndAdjointSolversWork();
  result += test.systemReducedGradientChecks();

  return result.summary();
}
