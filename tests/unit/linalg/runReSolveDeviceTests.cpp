#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <TestHelper.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaJacobian.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/native/HostJacobian.hpp>
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
  HostVector<Index> row_ptr(nx * ny + 1, 0);
  HostVector<Index> cols;
  for (Index y = 0; y < ny; ++y)
  {
    for (Index x = 0; x < nx; ++x)
    {
      HostVector<Index> row_cols;
      const Index       row = gridNode(x, y, nx);
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

HostVector<Real> expectedGridSolution(Index nx, Index ny)
{
  HostVector<Real> sol(nx * ny);
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

HostVector<Real> mul(const HostCsrMatrix& mat, const HostVector<Real>& x)
{
  HostVector<Real> out(mat.rows());
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

bool vecNear(const HostVector<Real>& actual, const HostVector<Real>& expected)
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

void copyMatrix(const HostCsrMatrix& source,
                DeviceCsrMatrix&     destination,
                linalg::CudaContext& ctx)
{
  if (source.pattern().layoutId() != destination.pattern().layoutId())
  {
    throw std::runtime_error(
        "Test matrix copy requires matching CSR layouts");
  }
  ctx.vectors().copy(source.vals().view(), destination.vals().view());
}

TestOutcome unifiedResolveSolvesDeviceStorage()
{
  TestStatus status(__func__);
  if (!linalg::CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
#if defined(FEMX_RESOLVE_BUILD_ID)
    std::cout << "    ReSolve build: " << FEMX_RESOLVE_BUILD_ID << '\n';
#endif

    constexpr Index      nx      = 16;
    constexpr Index      ny      = 16;
    const HostCsrPattern h_graph = gridGraph(nx, ny);
    HostCsrMatrix        h_mat(h_graph);
    fillGridMat(h_mat);
    HostCsrMatrix h_mat_tr_source(h_graph);
    fillGridMat(h_mat_tr_source);

    linalg::HostContext  cpu_ctx;
    linalg::CudaContext  ctx;
    linalg::HostJacobian h_jacobian(cpu_ctx);
    auto&                vec_handler = ctx.vectors();
    DeviceCsrPattern     d_graph;
    copy(h_graph, d_graph, ctx);

    DeviceCsrMatrix d_mat(d_graph);
    copyMatrix(h_mat, d_mat, ctx);
    DeviceCsrMatrix d_mat_tr_source(d_graph);
    copyMatrix(h_mat_tr_source, d_mat_tr_source, ctx);
    linalg::ReSolveLinearSolver solver;
    linalg::ReSolveLinearSolver tr_solver;

    const HostVector<Real> expected = expectedGridSolution(nx, ny);
    const HostVector<Real> h_rhs    = mul(h_mat, expected);
    DeviceVector<Real>     d_rhs;
    DeviceVector<Real>     d_sol;
    vec_handler.copy(h_rhs, d_rhs);

    bool alias_rejected = false;
    try
    {
      solver.solve(d_mat, d_rhs, d_rhs, ctx);
    }
    catch (const std::runtime_error&)
    {
      alias_rejected = true;
    }
    status *= alias_rejected;

    solver.solve(d_mat, d_rhs, d_sol, ctx);
    HostVector<Real> fwd_sol;
    vec_handler.copy(d_sol, fwd_sol);
    ctx.sync();
    status *= vecNear(fwd_sol, expected);

    HostVector<Real> tr_rhs(h_mat_tr_source.cols());
    h_jacobian.applyT(h_mat_tr_source,
                      expected.view(),
                      tr_rhs.view());

    DeviceVector<Real> d_tr_rhs;
    DeviceVector<Real> d_tr_sol;
    vec_handler.copy(tr_rhs, d_tr_rhs);

    bool tr_alias_rejected = false;
    try
    {
      tr_solver.solveT(d_mat_tr_source, d_tr_rhs, d_tr_rhs, ctx);
    }
    catch (const std::runtime_error&)
    {
      tr_alias_rejected = true;
    }
    status *= tr_alias_rejected;

    tr_solver.solveT(d_mat_tr_source, d_tr_rhs, d_tr_sol, ctx);
    HostVector<Real> h_tr_sol;
    vec_handler.copy(d_tr_sol, h_tr_sol);
    ctx.sync();
    status *= vecNear(h_tr_sol, expected);

    const Real*  source_vals    = d_mat.valsData();
    const Index* source_rows    = d_mat.rowPtrData();
    const Index* source_cols    = d_mat.colIndData();
    const Real*  tr_source_vals = d_mat_tr_source.valsData();
    const Real*  tr_sol_data    = d_tr_sol.data();

    HostVector<Real> zero_rhs(h_rhs.size(), 0.0);
    vec_handler.copy(zero_rhs, d_rhs);
    solver.solve(d_mat, d_rhs, d_sol, ctx);
    vec_handler.copy(d_sol, fwd_sol);
    ctx.sync();
    status *= vecNear(fwd_sol, zero_rhs);

    // Update vals in the same femx allocations and solve again. Give the
    // transpose source a different shift so the explicitly transposed matrix,
    // rather than the bound forward matrix, must be authoritative.
    fillGridMat(h_mat, 0.25);
    fillGridMat(h_mat_tr_source, 0.5);
    copyMatrix(h_mat, d_mat, ctx);
    copyMatrix(h_mat_tr_source, d_mat_tr_source, ctx);
    const HostVector<Real> rhs2 = mul(h_mat, expected);
    HostVector<Real>       tr_rhs2(h_mat_tr_source.cols());
    h_jacobian.applyT(h_mat_tr_source,
                      expected.view(),
                      tr_rhs2.view());
    vec_handler.copy(rhs2, d_rhs);
    vec_handler.copy(tr_rhs2, d_tr_rhs);
    solver.solve(d_mat, d_rhs, d_sol, ctx);
    tr_solver.solveT(d_mat_tr_source, d_tr_rhs, d_tr_sol, ctx);
    vec_handler.copy(d_sol, fwd_sol);
    vec_handler.copy(d_tr_sol, h_tr_sol);
    ctx.sync();
    status *= vecNear(fwd_sol, expected);
    status *= vecNear(h_tr_sol, expected);
    status *= source_vals == d_mat.valsData();
    status *= source_rows == d_mat.rowPtrData();
    status *= source_cols == d_mat.colIndData();
    status *= tr_source_vals == d_mat_tr_source.valsData();
    status *= tr_sol_data == d_tr_sol.data();
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
