#include <iostream>
#include <stdexcept>

#include <femx/problem/Residual.hpp>
#include <femx/solve/Newton.hpp>
#include <femx/algebra/DenseLinearSolver.hpp>
#include <femx/algebra/backends/native/DenseSystemMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearProblem final : public problem::Residual
{
public:
  problem::Dimensions dimensions() const override
  {
    return {2, 1, 2};
  }

  void residual(const Vector<Real>& state,
                const Vector<Real>& prm,
                Vector<Real>&       out) const override
  {
    resize(out, 2);
    out[0] = 2.0 * state[0] + 3.0 * state[1] + 5.0 * prm[0] - 1.0;
    out[1] = 7.0 * state[0] + 11.0 * state[1] + 13.0 * prm[0] + 2.0;
  }

  void linearize(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 problem::Linearization& out) const override
  {
    (void) state;
    (void) prm;

    auto* matrix_out = dynamic_cast<problem::MatrixLinearization*>(&out);
    if (matrix_out == nullptr)
    {
      throw std::runtime_error(
          "LinearProblem requires problem::MatrixLinearization");
    }

    algebra::MatrixOperator& state_jac = matrix_out->stateMatrix();
    state_jac.resize(2, 2);
    state_jac.setZero();
    state_jac.set(0, 0, 2.0);
    state_jac.set(0, 1, 3.0);
    state_jac.set(1, 0, 7.0);
    state_jac.set(1, 1, 11.0);
    state_jac.finalize();

    algebra::MatrixOperator& param_jac = matrix_out->paramMatrix();
    param_jac.resize(2, 1);
    param_jac.setZero();
    param_jac.set(0, 0, 5.0);
    param_jac.set(1, 0, 13.0);
    param_jac.finalize();
  }

private:
  static void resize(Vector<Real>& out, Index size)
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
};

class NewtonTests : public TestBase
{
public:
  TestOutcome solvesLinearProblem()
  {
    TestStatus status;
    status = true;

    LinearProblem             problem;
    algebra::DenseSystemMatrix state_jac;
    algebra::DenseSystemMatrix param_jac;
    problem::MatrixLinearization linearization(state_jac, param_jac);
    algebra::DenseLinearSolver    linear_solver;
    solve::Newton                newton(problem, linearization, linear_solver);

    status *= (newton.numStates() == 2);
    status *= (newton.numParams() == 1);
    status *= (newton.numResiduals() == 2);

    Vector<Real> prm(1);
    prm[0] = 2.0;

    Vector<Real> state;
    newton.solve(prm, state);

    status *= (state.size() == 2);
    status *= isEqual(state[0], -15.0);
    status *= isEqual(state[1], 7.0);

    Vector<Real> res;
    problem.residual(state, prm, res);
    status *= isEqual(res[0], 0.0);
    status *= isEqual(res[1], 0.0);

    return status.report(__func__);
  }

  TestOutcome honorsInitialState()
  {
    TestStatus status;
    status = true;

    LinearProblem             problem;
    algebra::DenseSystemMatrix state_jac;
    algebra::DenseSystemMatrix param_jac;
    problem::MatrixLinearization linearization(state_jac, param_jac);
    algebra::DenseLinearSolver    linear_solver;
    solve::Newton                newton(problem, linearization, linear_solver);

    Vector<Real> init(2);
    init[0] = 100.0;
    init[1] = -100.0;
    newton.setInitialState(init);

    Vector<Real> prm(1);
    prm[0] = 2.0;

    Vector<Real> state;
    newton.solve(prm, state);

    status *= isEqual(state[0], -15.0);
    status *= isEqual(state[1], 7.0);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running Newton tests:\n";

  femx::tests::NewtonTests test;

  femx::tests::TestingResults result;
  result += test.solvesLinearProblem();
  result += test.honorsInitialState();

  return result.summary();
}
