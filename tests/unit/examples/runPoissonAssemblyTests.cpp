#include <algorithm>
#include <cmath>

#include "PoissonProblem.hpp"
#include "PoissonResidual.hpp"
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

  examples::poisson::Options opts;
  opts.num_x_cells = 2;
  opts.num_y_cells = 2;
  examples::poisson::PoissonProblem      problem(opts);
  examples::poisson::HostPoissonResidual poisson_res(problem);

  linalg::HostLinearSystem     system;
  state::HostLinearStateSolver solver(poisson_res, system);
  const HostVector<Real>       prm;
  HostVector<Real>             state;
  solver.solve(state);

  auto& jac =
      dynamic_cast<linalg::HostJacobian&>(system.jacobian());
  const HostCsrMatrix& mat = jac.matrix();

  status *= mat.pattern().layoutId()
            == problem.assemblyMap().pattern().layoutId();
  status *= state.size() == problem.numDofs();
  status *= problem.mesh().numElems() == 4;

  const auto& map  = problem.boundaryMap();
  const auto& vals = problem.boundaryValues();
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

  HostVector<Real> res;
  poisson_res.assembleResidual(state, prm, res, system.context());
  bool has_positive_interior = false;
  for (Index row = 0; row < state.size(); ++row)
  {
    status *= std::isfinite(state[row]);
    status *= near(res[row], 0.0);
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
