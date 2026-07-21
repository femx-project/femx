#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "TestHelper.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>
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

HostCsrPattern gridGraph(Index nx, Index ny)
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
#if defined(FEMX_RESOLVE_BUILD_ID)
    std::cout << "    ReSolve build: " << FEMX_RESOLVE_BUILD_ID << '\n';
#endif

    constexpr Index      nx     = 16;
    constexpr Index      ny     = 16;
    const HostCsrPattern hgraph = gridGraph(nx, ny);
    HostCsrMatrix        hmat(hgraph);
    fillGridMat(hmat);
    HostCsrMatrix hmat_tr_source(hgraph);
    fillGridMat(hmat_tr_source);

    CpuContext                cpu_ctx;
    CudaContext               ctx;
    linalg::HostMatrixHandler host_mat_handler(cpu_ctx);
    linalg::CudaVectorHandler vec_handler(ctx);
    linalg::CudaMatrixHandler mat_handler(ctx);
    DeviceCsrPattern          dgraph;
    copy(hgraph, dgraph, ctx);

    DeviceCsrMatrix dmat(dgraph);
    mat_handler.copy(hmat, dmat);
    DeviceCsrMatrix dmat_tr_source(dgraph);
    mat_handler.copy(hmat_tr_source, dmat_tr_source);
    linalg::ReSolveLinearSolver solver;
    linalg::ReSolveLinearSolver tr_solver;

    const HostVector expected = expectedGridSolution(nx, ny);
    const HostVector hrhs     = mul(hmat, expected);
    DeviceVector     drhs;
    DeviceVector     dsol;
    vec_handler.copy(hrhs, drhs);

    bool alias_rejected = false;
    try
    {
      solver.solve(dmat, drhs, drhs, ctx);
    }
    catch (const std::runtime_error&)
    {
      alias_rejected = true;
    }
    status *= alias_rejected;

    solver.solve(dmat, drhs, dsol, ctx);
    HostVector fwd_sol;
    vec_handler.copy(dsol, fwd_sol);
    ctx.sync();
    status *= vecNear(fwd_sol, expected);

    HostVector tr_rhs(hmat_tr_source.cols());
    host_mat_handler.matvecT(hmat_tr_source,
                             expected.view(),
                             tr_rhs.view());

    DeviceVector dtr_rhs;
    DeviceVector dtr_sol;
    vec_handler.copy(tr_rhs, dtr_rhs);

    bool tr_alias_rejected = false;
    try
    {
      tr_solver.solveT(dmat_tr_source, dtr_rhs, dtr_rhs, ctx);
    }
    catch (const std::runtime_error&)
    {
      tr_alias_rejected = true;
    }
    status *= tr_alias_rejected;

    tr_solver.solveT(dmat_tr_source, dtr_rhs, dtr_sol, ctx);
    HostVector device_tr_sol;
    vec_handler.copy(dtr_sol, device_tr_sol);
    ctx.sync();
    status *= vecNear(device_tr_sol, expected);

    const Real*  source_vals    = dmat.valsData();
    const Index* source_rows    = dmat.rowPtrData();
    const Index* source_cols    = dmat.colIndData();
    const Real*  tr_source_vals = dmat_tr_source.valsData();
    const Real*  tr_sol_data    = dtr_sol.data();

    HostVector zero_rhs(hrhs.size(), 0.0);
    vec_handler.copy(zero_rhs, drhs);
    solver.solve(dmat, drhs, dsol, ctx);
    vec_handler.copy(dsol, fwd_sol);
    ctx.sync();
    status *= vecNear(fwd_sol, zero_rhs);

    // Update vals in the same femx allocations and solve again. Give the
    // transpose source a different shift so the explicitly transposed matrix,
    // rather than the bound forward matrix, must be authoritative.
    fillGridMat(hmat, 0.25);
    fillGridMat(hmat_tr_source, 0.5);
    mat_handler.copy(hmat, dmat);
    mat_handler.copy(hmat_tr_source, dmat_tr_source);
    const HostVector rhs2 = mul(hmat, expected);
    HostVector       tr_rhs2(hmat_tr_source.cols());
    host_mat_handler.matvecT(hmat_tr_source,
                             expected.view(),
                             tr_rhs2.view());
    vec_handler.copy(rhs2, drhs);
    vec_handler.copy(tr_rhs2, dtr_rhs);
    solver.solve(dmat, drhs, dsol, ctx);
    tr_solver.solveT(dmat_tr_source, dtr_rhs, dtr_sol, ctx);
    vec_handler.copy(dsol, fwd_sol);
    vec_handler.copy(dtr_sol, device_tr_sol);
    ctx.sync();
    status *= vecNear(fwd_sol, expected);
    status *= vecNear(device_tr_sol, expected);
    status *= source_vals == dmat.valsData();
    status *= source_rows == dmat.rowPtrData();
    status *= source_cols == dmat.colIndData();
    status *= tr_source_vals == dmat_tr_source.valsData();
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
