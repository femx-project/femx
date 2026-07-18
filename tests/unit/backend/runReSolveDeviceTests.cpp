#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "TestHelper.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/CsrTranspose.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>

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

TestOutcome unifiedResolveSolvesDeviceStorage()
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
    HostCsrMatrix hmat_tr_source(hgraph);
    fillGridMat(hmat_tr_source);
    const HostCsrTransposeMap tr_map(hgraph);
    HostCsrMatrix             hmat_tr(tr_map.trGraph());
    trVals(hmat_tr_source, tr_map, hmat_tr);

    CudaContext    ctx;
    DeviceCsrGraph dgraph;
    copy(hgraph, dgraph, ctx);

    DeviceCsrMatrix dmat(dgraph);
    copy(hmat, dmat, ctx);
    DeviceCsrMatrix dmat_tr_source(dgraph);
    copy(hmat_tr_source, dmat_tr_source, ctx);
    DeviceCsrTransposeMap dtr_map;
    copy(tr_map, dgraph, dtr_map, ctx);
    DeviceCsrMatrix dmat_tr(dtr_map.trGraph());
    trVals(dmat_tr_source, dtr_map, dmat_tr, ctx);

    linalg::ReSolveLinearSolver solver;
    solver.setOperator(dmat);
    linalg::ReSolveLinearSolver tr_solver;
    tr_solver.setOperator(dmat_tr);

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

    const HostVector            tr_rhs = mul(hmat_tr, expected);
    HostVector                  cpu_tr_sol;
    linalg::ReSolveLinearSolver cpu_tr_solver;
    cpu_tr_solver.setOperator(hmat_tr);
    cpu_tr_solver.solve(tr_rhs, cpu_tr_sol);

    DeviceVector dtr_rhs;
    DeviceVector dtr_sol;
    copy(tr_rhs, dtr_rhs, ctx);

    bool setup_required = false;
    try
    {
      linalg::ReSolveLinearSolver unbound_solver;
      unbound_solver.solve(dtr_rhs, dtr_sol, ctx);
    }
    catch (const std::runtime_error&)
    {
      setup_required = true;
    }
    status *= setup_required;

    bool tr_alias_rejected = false;
    try
    {
      tr_solver.solve(dtr_rhs, dtr_rhs, ctx);
    }
    catch (const std::runtime_error&)
    {
      tr_alias_rejected = true;
    }
    status *= tr_alias_rejected;

    tr_solver.solve(dtr_rhs, dtr_sol, ctx);
    HostVector device_tr_sol;
    copy(dtr_sol, device_tr_sol, ctx);
    ctx.synchronize();
    status *= vecNear(cpu_tr_sol, expected);
    status *= vecNear(device_tr_sol, cpu_tr_sol);

    const Real*  source_vals    = dmat.valsData();
    const Index* source_rows    = dmat.rowPtrData();
    const Index* source_cols    = dmat.colIndData();
    const Real*  tr_source_vals = dmat_tr_source.valsData();
    const Real*  tr_vals        = dmat_tr.valsData();
    const Index* tr_rows        = dmat_tr.rowPtrData();
    const Index* tr_cols        = dmat_tr.colIndData();
    const Real*  tr_sol_data    = dtr_sol.data();

    HostVector zero_rhs(hrhs.size(), 0.0);
    copy(zero_rhs, drhs, ctx);
    solver.solve(drhs, dsol, ctx);
    copy(dsol, fwd_sol, ctx);
    ctx.synchronize();
    status *= vecNear(fwd_sol, zero_rhs);

    // Update vals in the same femx allocations and solve again. Give the
    // transpose source a different shift so the explicitly transposed matrix,
    // rather than the bound forward matrix, must be authoritative.
    fillGridMat(hmat, 0.25);
    fillGridMat(hmat_tr_source, 0.5);
    trVals(hmat_tr_source, tr_map, hmat_tr);
    copy(hmat, dmat, ctx);
    copy(hmat_tr_source, dmat_tr_source, ctx);
    trVals(dmat_tr_source, dtr_map, dmat_tr, ctx);
    const HostVector rhs2    = mul(hmat, expected);
    const HostVector tr_rhs2 = mul(hmat_tr, expected);
    copy(rhs2, drhs, ctx);
    copy(tr_rhs2, dtr_rhs, ctx);
    solver.setOperator(dmat);
    tr_solver.setOperator(dmat_tr);
    solver.solve(drhs, dsol, ctx);
    tr_solver.solve(dtr_rhs, dtr_sol, ctx);
    cpu_tr_solver.setOperator(hmat_tr);
    cpu_tr_solver.solve(tr_rhs2, cpu_tr_sol);
    copy(dsol, fwd_sol, ctx);
    copy(dtr_sol, device_tr_sol, ctx);
    ctx.synchronize();
    status *= vecNear(fwd_sol, expected);
    status *= vecNear(device_tr_sol, cpu_tr_sol);
    status *= vecNear(device_tr_sol, expected);
    status *= source_vals == dmat.valsData();
    status *= source_rows == dmat.rowPtrData();
    status *= source_cols == dmat.colIndData();
    status *= tr_source_vals == dmat_tr_source.valsData();
    status *= tr_vals == dmat_tr.valsData();
    status *= tr_rows == dmat_tr.rowPtrData();
    status *= tr_cols == dmat_tr.colIndData();
    status *= tr_sol_data == dtr_sol.data();
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
  results += femx::tests::unifiedResolveSolvesDeviceStorage();
  return results.summary();
}
