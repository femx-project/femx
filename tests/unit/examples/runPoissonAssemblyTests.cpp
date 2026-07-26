#include <algorithm>
#include <cmath>

#include "PoissonForward.hpp"
#include "TestHelper.hpp"
#include <femx/linalg/native/HostJacobian.hpp>
#include <femx/linalg/native/HostLinearSystem.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx::tests
{
namespace
{

bool near(Real lhs, Real rhs)
{
  return std::abs(lhs - rhs) <= 1.0e-12;
}

TestOutcome poissonUsesMappedGraphAndBoundaryRows()
{
  TestStatus status(__func__);

  examples::poisson::Options options;
  options.num_x_cells = 2;
  options.num_y_cells = 2;
  examples::poisson::PoissonForwardProblem problem(options);

  linalg::HostLinearSystem     system;
  state::HostLinearStateSolver solver(problem, system);
  const HostVector<Real>       prm;
  HostVector<Real>             state;
  solver.solve(prm, state);

  auto& jacobian =
      dynamic_cast<linalg::HostJacobian&>(system.jacobian());
  const HostCsrMatrix& mat = jacobian.matrix();

  status *= mat.pattern().layoutId()
            == problem.map().pattern().layoutId();
  status *= state.size() == problem.numDofs();
  status *= problem.geom().numElems() == 4;

  const auto& map  = problem.bcMap();
  const auto& vals = problem.bcVals();
  const auto  view = map.view();
  for (Index ib = 0; ib < map.numBcs(); ++ib)
  {
    const Index row  = view.constrained_rows[ib];
    status          *= near(state[row], vals[ib]);
    for (Index k = mat.rowPtrData()[row]; k < mat.rowPtrData()[row + 1]; ++k)
    {
      const Real expected  = mat.colIndData()[k] == row ? 1.0 : 0.0;
      status              *= near(mat.valsData()[k], expected);
    }
  }

  HostVector<Real> residual;
  problem.res(state, prm, residual, system.context());
  bool has_positive_interior = false;
  for (Index row = 0; row < state.size(); ++row)
  {
    status *= std::isfinite(state[row]);
    status *= near(residual[row], 0.0);
    const bool constrained =
        std::find(view.constrained_rows.begin(),
                  view.constrained_rows.end(),
                  row)
        != view.constrained_rows.end();
    if (!constrained && state[row] > 0.0)
    {
      has_positive_interior = true;
    }
  }
  status *= has_positive_interior;

  return status.report();
}

} // namespace
} // namespace femx::tests

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::poissonUsesMappedGraphAndBoundaryRows();
  return results.summary();
}
