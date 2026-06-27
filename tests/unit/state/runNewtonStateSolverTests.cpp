#include <cmath>
#include <stdexcept>

#include <femx/linalg/native/DenseLinearSolver.hpp>
#include <femx/linalg/native/DenseMatrixOperator.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/state/NewtonStateSolver.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{
namespace
{

class QuadraticResidual final : public problem::Residual
{
public:
  problem::Dimensions dims() const override
  {
    return {1, 1, 1};
  }

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    resizeOrZero(out, 1);
    out[0] = state[0] * state[0] - prm[0];
  }

  void linearize(const Vector<Real>&     state,
                 const Vector<Real>&     prm,
                 problem::Linearization& out) const override
  {
    (void) prm;
    auto* matrix_out = dynamic_cast<problem::MatrixLinearization*>(&out);
    if (matrix_out == nullptr)
    {
      throw std::runtime_error(
          "QuadraticResidual requires MatrixLinearization");
    }

    matrix_out->stateMatrix().resize(1, 1);
    matrix_out->stateMatrix().setZero();
    matrix_out->stateMatrix().set(0, 0, 2.0 * state[0]);
    matrix_out->stateMatrix().finalize();

    matrix_out->paramMatrix().resize(1, 1);
    matrix_out->paramMatrix().setZero();
    matrix_out->paramMatrix().set(0, 0, -1.0);
    matrix_out->paramMatrix().finalize();
  }
};

bool near(Real actual, Real expected, Real tol)
{
  return std::abs(actual - expected) <= tol;
}

} // namespace

class NewtonStateSolverTests
{
public:
  TestOutcome solvesStationaryNonlinearProblem()
  {
    TestStatus status;
    status = true;

    QuadraticResidual            problem;
    linalg::DenseMatrixOperator  J_state;
    linalg::DenseMatrixOperator  J_param;
    problem::MatrixLinearization lin(J_state, J_param);
    linalg::DenseLinearSolver    lin_solver;
    state::NewtonStateSolver     solver(problem, lin, lin_solver);
    solver.opts().residual_tolerance = 1.0e-13;

    Vector<Real> initial(1);
    initial[0] = 1.0;
    solver.setInitialState(initial);

    Vector<Real> prm(1);
    prm[0] = 2.25;

    Vector<Real>        state;
    state::StateSolver& base = solver;
    base.solve(prm, state);
    status *= state.size() == 1;
    status *= near(state[0], 1.5, 1.0e-12);

    problem.linearize(state, prm, lin);
    Vector<Real> dir(1);
    dir[0] = 1.0;
    Vector<Real> applied;
    lin.stateJac().apply(dir, applied);
    status *= applied.size() == 1;
    status *= near(applied[0], 3.0, 1.0e-12);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults         results;
  femx::tests::NewtonStateSolverTests test;

  results += test.solvesStationaryNonlinearProblem();

  return results.summary();
}
