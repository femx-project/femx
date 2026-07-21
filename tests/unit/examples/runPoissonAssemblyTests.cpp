#include <cmath>

#include "PoissonForward.hpp"
#include "TestHelper.hpp"

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

  HostCsrMatrix mat(problem.map().pattern());
  HostVector    rhs;
  problem.assemble(mat, rhs);

  status *= mat.pattern().layoutId()
            == problem.map().pattern().layoutId();
  status *= rhs.size() == problem.numDofs();
  status *= problem.geom().numElems() == 4;

  const auto& map  = problem.bcMap();
  const auto& vals = problem.bcVals();
  const auto  view = map.view();
  for (Index ib = 0; ib < map.numBcs(); ++ib)
  {
    const Index row  = view.bcRow(ib);
    status          *= near(rhs[row], vals[ib]);
    for (Index k = mat.rowPtrData()[row]; k < mat.rowPtrData()[row + 1]; ++k)
    {
      const Real expected  = mat.colIndData()[k] == row ? 1.0 : 0.0;
      status              *= near(mat.valsData()[k], expected);
    }
  }

  bool has_interior_forcing = false;
  for (Index row = 0; row < rhs.size(); ++row)
  {
    status *= std::isfinite(rhs[row]);
    if (!view.isBc(row) && rhs[row] > 0.0)
    {
      has_interior_forcing = true;
    }
  }
  status *= has_interior_forcing;

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
