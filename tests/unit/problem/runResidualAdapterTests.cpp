#include <iostream>

#include <femx/problem/ProblemResidualAdapter.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/algebra/backends/native/DenseSystemMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearMatrixResidualEquation final : public problem::MatrixResidualEquation
{
public:
  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 1;
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
    out[0] = 2.0 * state[0] + 3.0 * state[1] + 5.0 * prm[0];
    out[1] = 7.0 * state[0] + 11.0 * state[1] + 13.0 * prm[0];
  }

  void assembleStateJac(const Vector<Real>&   state,
                        const Vector<Real>&   prm,
                        algebra::SystemMatrix& out) const override
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
                        algebra::SystemMatrix& out) const override
  {
    (void) state;
    (void) prm;
    out.resize(numRes(), numParams());
    out.setZero();
    out.set(0, 0, 5.0);
    out.set(1, 0, 13.0);
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

class ResidualAdapterTests : public TestBase
{
public:
  TestOutcome operatorAdapterExposesLinearization()
  {
    TestStatus status;
    status = true;

    LinearMatrixResidualEquation eq;
    problem::ResidualEquationProblemAdapter adapter(eq);

    const problem::Dimensions dims = adapter.dimensions();
    status *= (dims.num_states == 2);
    status *= (dims.num_params == 1);
    status *= (dims.num_residuals == 2);

    Vector<Real> state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector<Real> prm(1);
    prm[0] = 2.0;

    Vector<Real> out;
    adapter.residual(state, prm, out);
    status *= isEqual(out[0], 2.0 * state[0] + 3.0 * state[1] + 5.0 * prm[0]);
    status *= isEqual(out[1], 7.0 * state[0] + 11.0 * state[1] + 13.0 * prm[0]);

    problem::ResidualEquationLinearization lin;
    adapter.linearize(state, prm, lin);

    Vector<Real> state_dir(2);
    state_dir[0] = 1.5;
    state_dir[1] = -2.0;

    lin.stateJacobian().apply(state_dir, out);
    status *= isEqual(out[0], 2.0 * state_dir[0] + 3.0 * state_dir[1]);
    status *= isEqual(out[1], 7.0 * state_dir[0] + 11.0 * state_dir[1]);

    Vector<Real> lambda(2);
    lambda[0] = -3.0;
    lambda[1] = 4.0;

    lin.stateJacobian().applyT(lambda, out);
    status *= isEqual(out[0], 2.0 * lambda[0] + 7.0 * lambda[1]);
    status *= isEqual(out[1], 3.0 * lambda[0] + 11.0 * lambda[1]);

    Vector<Real> param_dir(1);
    param_dir[0] = -0.75;

    lin.paramJacobian().apply(param_dir, out);
    status *= isEqual(out[0], 5.0 * param_dir[0]);
    status *= isEqual(out[1], 13.0 * param_dir[0]);

    lin.paramJacobian().applyT(lambda, out);
    status *= isEqual(out[0], 5.0 * lambda[0] + 13.0 * lambda[1]);

    return status.report(__func__);
  }

  TestOutcome matrixAdapterAssemblesLinearization()
  {
    TestStatus status;
    status = true;

    LinearMatrixResidualEquation eq;
    problem::MatrixResidualEquationProblemAdapter adapter(eq);

    Vector<Real> state(2);
    state[0] = -1.0;
    state[1] = 0.5;

    Vector<Real> prm(1);
    prm[0] = 3.0;

    algebra::DenseSystemMatrix state_jac;
    algebra::DenseSystemMatrix param_jac;
    problem::MatrixLinearization lin(state_jac, param_jac);
    adapter.linearize(state, prm, lin);

    Vector<Real> out;
    Vector<Real> state_dir(2);
    state_dir[0] = 4.0;
    state_dir[1] = -1.0;

    lin.stateJacobian().apply(state_dir, out);
    status *= isEqual(out[0], 2.0 * state_dir[0] + 3.0 * state_dir[1]);
    status *= isEqual(out[1], 7.0 * state_dir[0] + 11.0 * state_dir[1]);

    Vector<Real> lambda(2);
    lambda[0] = 2.5;
    lambda[1] = -0.25;

    lin.stateJacobian().applyT(lambda, out);
    status *= isEqual(out[0], 2.0 * lambda[0] + 7.0 * lambda[1]);
    status *= isEqual(out[1], 3.0 * lambda[0] + 11.0 * lambda[1]);

    Vector<Real> param_dir(1);
    param_dir[0] = -2.0;

    lin.paramJacobian().apply(param_dir, out);
    status *= isEqual(out[0], 5.0 * param_dir[0]);
    status *= isEqual(out[1], 13.0 * param_dir[0]);

    lin.paramJacobian().applyT(lambda, out);
    status *= isEqual(out[0], 5.0 * lambda[0] + 13.0 * lambda[1]);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running residual adapter tests:\n";

  femx::tests::ResidualAdapterTests test;

  femx::tests::TestingResults result;
  result += test.operatorAdapterExposesLinearization();
  result += test.matrixAdapterAssemblesLinearization();

  return result.summary();
}
