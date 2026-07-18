#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "TestHelper.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/resolve/ReSolveDeviceSolver.hpp>

namespace femx
{
namespace tests
{
namespace
{

bool near(Real lhs, Real rhs, Real tolerance = 1.0e-6)
{
  return std::abs(lhs - rhs) <= tolerance;
}

Index gridNode(Index x, Index y, Index nx)
{
  return y * nx + x;
}

HostCsrGraph gridGraph(Index nx, Index ny)
{
  HostIndexVector row_ptr(nx * ny + 1, 0);
  HostIndexVector cols;
  for (Index y = 0; y < ny; ++y)
  {
    for (Index x = 0; x < nx; ++x)
    {
      Array<Index> row_cols;
      const Index  row = gridNode(x, y, nx);
      row_cols.push_back(row);
      if (x > 0)
        row_cols.push_back(gridNode(x - 1, y, nx));
      if (x + 1 < nx)
        row_cols.push_back(gridNode(x + 1, y, nx));
      if (y > 0)
        row_cols.push_back(gridNode(x, y - 1, nx));
      if (y + 1 < ny)
        row_cols.push_back(gridNode(x, y + 1, nx));
      std::sort(row_cols.begin(), row_cols.end());
      for (Index col : row_cols)
      {
        cols.push_back(col);
      }
      row_ptr[row + 1] = cols.size();
    }
  }
  return {nx * ny,
          nx * ny,
          std::move(row_ptr),
          std::move(cols)};
}

void fillGridMat(HostCsrMatrix& mat, Real diag_shift = 0.0)
{
  for (Index row = 0; row < mat.rows(); ++row)
  {
    for (Index k = mat.rowPtrData()[row];
         k < mat.rowPtrData()[row + 1];
         ++k)
    {
      const Index col = mat.colIndData()[k];
      if (row == col)
      {
        mat.valsData()[k] = 4.0 + diag_shift;
      }
      else
      {
        mat.valsData()[k] = col > row ? -1.1 : -0.9;
      }
    }
  }
}

HostVector expectedGridSolution(Index nx, Index ny)
{
  HostVector sol(nx * ny);
  for (Index y = 0; y < ny; ++y)
  {
    for (Index x = 0; x < nx; ++x)
    {
      sol[gridNode(x, y, nx)] =
          0.5 + 0.01 * x - 0.02 * y + 0.001 * x * y;
    }
  }
  return sol;
}

HostVector mul(const HostCsrMatrix& mat, const HostVector& x)
{
  HostVector out(mat.rows());
  for (Index row = 0; row < mat.rows(); ++row)
  {
    for (Index k = mat.rowPtrData()[row];
         k < mat.rowPtrData()[row + 1];
         ++k)
    {
      out[row] += mat.valsData()[k]
                  * x[mat.colIndData()[k]];
    }
  }
  return out;
}

bool vecNear(const HostVector& actual, const HostVector& expected)
{
  if (actual.size() != expected.size())
  {
    return false;
  }
  for (Index i = 0; i < actual.size(); ++i)
  {
    if (!near(actual[i], expected[i]))
    {
      std::cout << "    mismatch at " << i << ": got " << actual[i]
                << ", expected " << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

TestOutcome resolveSolvesDeviceStorage()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
#if defined(FEMX_RESOLVE_DEVELOP_PREFIX) && defined(FEMX_RESOLVE_BUILD_ID)
    std::cout << "    ReSolve prefix: " << FEMX_RESOLVE_DEVELOP_PREFIX << '\n';
    std::cout << "    ReSolve build: " << FEMX_RESOLVE_BUILD_ID << '\n';
#endif

    constexpr Index    nx     = 16;
    constexpr Index    ny     = 16;
    const HostCsrGraph hgraph = gridGraph(nx, ny);
    HostCsrMatrix      hmat(hgraph);
    fillGridMat(hmat);

    CudaContext    ctx;
    DeviceCsrGraph dgraph;
    copy(hgraph, dgraph, ctx);

    DeviceCsrMatrix dmat(dgraph);
    copy(hmat, dmat, ctx);

    linalg::ReSolveOptions options;
    options.rtol    = 1.0e-10;
    options.max_its = 100;
    options.restart = 20;
    linalg::ReSolveDeviceSolver solver(options);
    solver.setOperator(dmat);

    const HostVector expected = expectedGridSolution(nx, ny);
    const HostVector hrhs     = mul(hmat, expected);
    DeviceVector     drhs;
    DeviceVector     dsol;
    copy(hrhs, drhs, ctx);

    bool alias_rejected = false;
    try
    {
      solver.solve(drhs, drhs, ctx);
    }
    catch (const std::runtime_error&)
    {
      alias_rejected = true;
    }
    status *= alias_rejected;

    solver.solve(drhs, dsol, ctx);
    HostVector fwd_sol;
    copy(dsol, fwd_sol, ctx);
    ctx.synchronize();
    status *= vecNear(fwd_sol, expected);

    HostVector zero_rhs(hrhs.size(), 0.0);
    copy(zero_rhs, drhs, ctx);
    solver.solve(drhs, dsol, ctx);
    copy(dsol, fwd_sol, ctx);
    ctx.synchronize();
    status *= vecNear(fwd_sol, zero_rhs);

    // Update vals in the same femx allocation and solve again. The ReSolve
    // mat and vec wrappers remain borrowed and persistent.
    fillGridMat(hmat, 0.25);
    copy(hmat, dmat, ctx);
    const HostVector rhs2 = mul(hmat, expected);
    copy(rhs2, drhs, ctx);
    solver.setOperator(dmat);
    solver.solve(drhs, dsol, ctx);
    copy(dsol, fwd_sol, ctx);
    ctx.synchronize();
    status *= vecNear(fwd_sol, expected);
  }
  catch (const std::exception& error)
  {
    std::cout << "    exception: " << error.what() << '\n';
    status *= false;
  }

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::resolveSolvesDeviceStorage();
  return results.summary();
}
