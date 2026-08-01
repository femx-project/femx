#include <cmath>
#include <exception>
#include <iostream>
#include <utility>

#include <TestHelper.hpp>
#include <femx/assembly/Assembly.hpp>
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>
#include <femx/linalg/host/HostContext.hpp>
#include <femx/linalg/host/HostSystemMatrix.hpp>

namespace femx
{
namespace tests
{
namespace
{

struct AffineElementKernel
{
  template <MemorySpace Space>
  FEMX_HOST_DEVICE void evalRow(
      const assembly::ElementView<Space>& in,
      Index                               row,
      Real&                               res,
      VectorView<Space, Real>             jac) const
  {
    res = in.state[row] + static_cast<Real>(in.ie + 1)
          + in.coords[0];
    for (Index col = 0; col < jac.size(); ++col)
    {
      jac[col] = row == col ? 2.0 : 1.0;
    }
  }
};

struct TimeElementKernel
{
  template <MemorySpace Space>
  FEMX_HOST_DEVICE void evalRow(
      const assembly::TimeElementView<Space>& elem,
      state::VariableBlock                    wrt,
      Index                                   row,
      Real&                                   res,
      VectorView<Space, Real>                 jac) const
  {
    res = elem.nxt[row] - 2.0 * elem.histState(0)[row]
          + 0.5 * elem.histState(1)[row]
          + static_cast<Real>(elem.ie + elem.step);
    const Real diag = wrt.isNextState()
                          ? 1.0
                          : (wrt.historyLag() == 0 ? -2.0 : 0.5);
    for (Index col = 0; col < jac.size(); ++col)
    {
      jac[col] = row == col ? diag : 0.0;
    }
  }
};

bool vecsNear(const HostVector<Real>& lhs,
              const HostVector<Real>& rhs,
              Real                    tolerance = 1.0e-12)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }
  for (Index i = 0; i < lhs.size(); ++i)
  {
    if (std::abs(lhs[i] - rhs[i]) > tolerance)
    {
      return false;
    }
  }
  return true;
}

bool matsNear(const HostCsrMatrix& lhs,
              const HostCsrMatrix& rhs,
              Real                 tolerance = 1.0e-12)
{
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()
      || lhs.nnz() != rhs.nnz())
  {
    return false;
  }
  for (Index k = 0; k < lhs.nnz(); ++k)
  {
    if (lhs.colIndData()[k] != rhs.colIndData()[k]
        || std::abs(lhs.valsData()[k] - rhs.valsData()[k])
               > tolerance)
    {
      return false;
    }
  }
  for (Index row = 0; row <= lhs.rows(); ++row)
  {
    if (lhs.rowPtrData()[row] != rhs.rowPtrData()[row])
    {
      return false;
    }
  }
  return true;
}

void recordCheck(TestStatus& status, bool condition, const char* label)
{
  if (!condition)
  {
    std::cout << "    failed check: " << label << '\n';
  }
  status *= condition;
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
  ctx.vectorHandler().copy(source.vals().view(), destination.vals().view());
}

void copyMatrix(const DeviceCsrMatrix& source,
                HostCsrMatrix&         destination,
                linalg::CudaContext&   ctx)
{
  if (source.pattern().layoutId() != destination.pattern().layoutId())
  {
    throw std::runtime_error(
        "Test matrix copy requires matching CSR layouts");
  }
  ctx.vectorHandler().copy(source.vals().view(), destination.vals().view());
}

void loadMatrix(const HostCsrMatrix&      source,
                linalg::HostSystemMatrix& destination)
{
  destination.setup(source.pattern());
  HostVector<Index> row(1);
  for (Index global_row = 0; global_row < source.rows(); ++global_row)
  {
    const Index begin = source.rowPtrData()[global_row];
    const Index count =
        source.rowPtrData()[global_row + 1] - begin;
    HostVector<Index> entries(count);
    for (Index i = 0; i < count; ++i)
    {
      entries[i] = begin + i;
    }
    row[0] = global_row;
    destination.addElement(
        {row.view(),
         {source.colIndData() + begin, count},
         entries.view(),
         {source.valsData() + begin, 1, count}});
  }
}

void loadMatrix(const HostCsrMatrix&      source,
                linalg::CudaSystemMatrix& destination,
                linalg::CudaContext&      ctx)
{
  destination.setup(source.pattern());
  ctx.vectorHandler().copy(source.vals().view(),
                           destination.assemblyView().values);
}

HostCsrPattern denseThreeByThreeGraph()
{
  return {3,
          3,
          HostVector<Index>{0, 3, 6, 9},
          HostVector<Index>{0, 1, 2, 0, 1, 2, 0, 1, 2}};
}

void setDenseVals(HostCsrMatrix& mat)
{
  mat.vals() = {4.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0};
}

TestOutcome cudaAssemblyMatchesCpuReference()
{
  TestStatus status(__func__);
  if (!linalg::CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    fem::Mesh           mesh = fem::Mesh::makeStructuredQuad(2, 1);
    fem::LagrangeQuadQ1 element;
    fem::FESpace        space(&mesh, &element);
    space.setup();

    const auto h_map =
        assembly::makeAssemblyMap(space.dofMap());
    const HostVector<Real> h_state{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    HostVector<Real>         cpu_res;
    linalg::HostContext      cpu_ctx;
    linalg::HostSystemMatrix cpu_jac_sys(cpu_ctx);
    cpu_jac_sys.setup(h_map.pattern());
    assembly::assembleResidualAndJacobian(AffineElementKernel{},
                                          mesh,
                                          h_map,
                                          h_state,
                                          cpu_res,
                                          cpu_jac_sys,
                                          cpu_ctx);
    const HostCsrMatrix& cpu_jac = cpu_jac_sys.matrix();

    linalg::CudaContext         cuda_ctx;
    auto&                       vec_handler = cuda_ctx.vectorHandler();
    fem::DeviceMesh             d_mesh;
    assembly::DeviceAssemblyMap d_map;
    DeviceVector<Real>          d_state;

    fem::copy(mesh, d_mesh, cuda_ctx);
    assembly::copy(h_map, d_map, cuda_ctx);
    vec_handler.copy(h_state, d_state);
    DeviceVector<Real> state_clone;
    vec_handler.copy(d_state, state_clone);

    DeviceVector<Real>       d_res;
    linalg::CudaSystemMatrix d_jac(cuda_ctx);
    d_jac.setup(h_map.pattern());
    auto moved_d_map = std::move(d_map);
    assembly::assembleResidualAndJacobian(AffineElementKernel{},
                                          d_mesh,
                                          moved_d_map,
                                          state_clone,
                                          d_res,
                                          d_jac,
                                          cuda_ctx);

    HostVector<Real> gpu_res;
    HostCsrMatrix    gpu_jac(h_map.pattern());
    vec_handler.copy(d_res, gpu_res);
    copyMatrix(d_jac.matrix(), gpu_jac, cuda_ctx);
    cuda_ctx.sync();

    recordCheck(status,
                vecsNear(gpu_res, cpu_res),
                "CUDA res matches CPU");
    recordCheck(status,
                matsNear(gpu_jac, cpu_jac),
                "CUDA Jacobian matches CPU");

    linalg::CudaSystemMatrix d_jac_only(cuda_ctx);
    d_jac_only.setup(h_map.pattern());
    assembly::assembleJacobian(AffineElementKernel{},
                               d_mesh,
                               moved_d_map,
                               state_clone,
                               d_jac_only,
                               cuda_ctx);
    HostCsrMatrix gpu_jac_only(h_map.pattern());
    copyMatrix(
        d_jac_only.matrix(), gpu_jac_only, cuda_ctx);
    cuda_ctx.sync();
    recordCheck(status,
                matsNear(gpu_jac_only, cpu_jac),
                "CUDA Jacobian-only assembly matches CPU");

    recordCheck(status,
                mesh.maxElemNodes() == 4,
                "mesh maximum element nodes");

    bool mat_alias_rejected = false;
    try
    {
      assembly::assembleResidualAndJacobian(AffineElementKernel{},
                                            d_mesh,
                                            moved_d_map,
                                            state_clone,
                                            state_clone,
                                            d_jac,
                                            cuda_ctx);
    }
    catch (const std::runtime_error&)
    {
      mat_alias_rejected = true;
    }
    recordCheck(status,
                mat_alias_rejected,
                "assembly rejects mat-val alias");
  }
  catch (const std::exception& error)
  {
    std::cout << "    exception: " << error.what() << '\n';
    status *= false;
  }

  return status.report();
}

TestOutcome cudaBoundaryMatchesCpuReference()
{
  TestStatus status(__func__);
  if (!linalg::CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    const HostCsrPattern h_graph = denseThreeByThreeGraph();
    const auto           h_map =
        assembly::makeBoundaryMap(HostVector<Index>{0, 2});

    HostCsrMatrix h_mat(h_graph);
    setDenseVals(h_mat);
    const HostVector<Real>   initial_rhs{10.0, 20.0, 30.0};
    HostVector<Real>         expected_rhs = initial_rhs;
    const HostVector<Real>   bc_vals{2.0, -1.0};
    linalg::HostContext      h_ctx;
    linalg::HostSystemMatrix expected_jac(h_ctx);
    loadMatrix(h_mat, expected_jac);
    expected_jac.eliminateColumns(h_map.view().constrained_rows,
                                  bc_vals.view(),
                                  expected_rhs.view());
    const HostCsrMatrix& expected_mat =
        expected_jac.matrix();

    linalg::CudaContext         ctx;
    auto&                       vec_handler = ctx.vectorHandler();
    assembly::DeviceBoundaryMap d_map;
    assembly::copy(h_map, d_map, ctx);

    linalg::CudaSystemMatrix d_jac(ctx);
    loadMatrix(h_mat, d_jac, ctx);
    DeviceVector<Real> d_rhs;
    DeviceVector<Real> d_bc;
    vec_handler.copy(initial_rhs, d_rhs);
    vec_handler.copy(bc_vals, d_bc);

    d_jac.eliminateColumns(
        d_map.view().constrained_rows, d_bc.view(), d_rhs.view());

    HostCsrMatrix    actual_mat(h_graph);
    HostVector<Real> actual_rhs;
    copyMatrix(d_jac.matrix(), actual_mat, ctx);
    vec_handler.copy(d_rhs, actual_rhs);
    ctx.sync();
    recordCheck(status,
                matsNear(actual_mat, expected_mat),
                "CUDA forward mat matches CPU");
    recordCheck(status,
                vecsNear(actual_rhs, expected_rhs),
                "CUDA forward RHS matches CPU");

    linalg::HostSystemMatrix expected_hist_jac(h_ctx);
    loadMatrix(h_mat, expected_hist_jac);
    expected_hist_jac.replaceRows(
        h_map.view().constrained_rows, 0.0);
    const HostCsrMatrix& expected_hist =
        expected_hist_jac.matrix();

    loadMatrix(h_mat, d_jac, ctx);
    d_jac.replaceRows(d_map.view().constrained_rows, 0.0);
    copyMatrix(d_jac.matrix(), actual_mat, ctx);
    ctx.sync();
    recordCheck(status,
                matsNear(actual_mat, expected_hist),
                "CUDA history rows match CPU");

    linalg::HostSystemMatrix expected_dirichlet_jac(h_ctx);
    loadMatrix(h_mat, expected_dirichlet_jac);
    assembly::applyDirichletConditions(
        h_map, expected_dirichlet_jac);
    const HostCsrMatrix& expected_dirichlet =
        expected_dirichlet_jac.matrix();

    loadMatrix(h_mat, d_jac, ctx);
    assembly::applyDirichletConditions(d_map, d_jac);
    copyMatrix(d_jac.matrix(), actual_mat, ctx);
    ctx.sync();
    recordCheck(status,
                matsNear(actual_mat, expected_dirichlet),
                "CUDA Dirichlet Jacobian matches CPU");

    const HostVector<Real> h_state{4.0, 5.0, 6.0};
    const HostVector<Real> h_res{10.0, 20.0, 30.0};
    DeviceVector<Real>     d_state;
    DeviceVector<Real>     d_res;
    vec_handler.copy(h_state, d_state);
    vec_handler.copy(h_res, d_res);
    assembly::applyDirichletConditions(d_map,
                                       d_state.view(),
                                       d_bc.view(),
                                       d_res.view(),
                                       ctx);
    HostVector<Real> actual_res;
    vec_handler.copy(d_res, actual_res);
    ctx.sync();
    recordCheck(status,
                vecsNear(actual_res,
                         HostVector<Real>{2.0, 20.0, 7.0}),
                "CUDA residual Dirichlet conditions");

    bool alias_rejected = false;
    try
    {
      assembly::applyDirichletConditions(d_map,
                                         d_state.view(),
                                         d_bc.view(),
                                         d_state.view(),
                                         ctx);
    }
    catch (const std::runtime_error&)
    {
      alias_rejected = true;
    }
    recordCheck(status,
                alias_rejected,
                "Residual Dirichlet conditions reject output alias");

    const HostCsrPattern different_layout{
        3,
        3,
        HostVector<Index>{0, 3, 6, 9},
        HostVector<Index>{1, 0, 2, 0, 2, 1, 2, 1, 0}};
    DeviceCsrPattern d_different_graph;
    femx::copy(different_layout, d_different_graph, ctx);
    DeviceCsrMatrix wrong_mat(d_different_graph);
    bool            layout_rejected = false;
    try
    {
      copyMatrix(h_mat, wrong_mat, ctx);
    }
    catch (const std::runtime_error&)
    {
      layout_rejected = true;
    }
    recordCheck(status,
                layout_rejected,
                "mat copy rejects a different layout");
  }
  catch (const std::exception& error)
  {
    std::cout << "    exception: " << error.what() << '\n';
    status *= false;
  }

  return status.report();
}

TestOutcome cudaTimeAssemblyMatchesCpuReference()
{
  TestStatus status(__func__);
  if (!linalg::CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    const auto map = assembly::makeAssemblyMap(
        3,
        3,
        HostVector<HostVector<Index>>{{0, 1}, {1, 2}},
        HostVector<HostVector<Index>>{{0, 1}, {1, 2}});
    const HostVector<Real>   hist{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const HostVector<Real>   nxt{7.0, 8.0, 9.0};
    HostVector<Real>         cpu_res;
    linalg::HostContext      cpu_ctx;
    linalg::HostSystemMatrix cpu_jac_sys(cpu_ctx);
    cpu_jac_sys.setup(map.pattern());
    assembly::assembleResidualAndJacobian(TimeElementKernel{},
                                          3,
                                          2,
                                          state::VariableBlock::hist(1),
                                          map,
                                          0,
                                          map.numElems(),
                                          hist.view(),
                                          nxt.view(),
                                          cpu_res,
                                          cpu_jac_sys,
                                          cpu_ctx);
    const HostCsrMatrix& cpu_jac = cpu_jac_sys.matrix();

    linalg::CudaContext         ctx;
    auto&                       vec_handler = ctx.vectorHandler();
    assembly::DeviceAssemblyMap d_map;
    DeviceVector<Real>          d_hist;
    DeviceVector<Real>          d_nxt;
    DeviceVector<Real>          d_res;
    assembly::copy(map, d_map, ctx);
    vec_handler.copy(hist, d_hist);
    vec_handler.copy(nxt, d_nxt);
    linalg::CudaSystemMatrix d_jac(ctx);
    d_jac.setup(map.pattern());
    assembly::assembleResidualAndJacobian(TimeElementKernel{},
                                          3,
                                          2,
                                          state::VariableBlock::hist(1),
                                          d_map,
                                          d_hist.view(),
                                          d_nxt.view(),
                                          d_res,
                                          d_jac,
                                          ctx);

    HostVector<Real> gpu_res;
    HostCsrMatrix    gpu_jac(map.pattern());
    vec_handler.copy(d_res, gpu_res);
    copyMatrix(d_jac.matrix(), gpu_jac, ctx);
    ctx.sync();
    recordCheck(status, vecsNear(gpu_res, cpu_res), "CUDA time res");
    recordCheck(status, matsNear(gpu_jac, cpu_jac), "CUDA time jac");

    assembly::assembleResidual(TimeElementKernel{},
                               3,
                               2,
                               d_map,
                               d_hist.view(),
                               d_nxt.view(),
                               d_res,
                               ctx);
    HostVector<Real> gpu_res_only;
    vec_handler.copy(d_res, gpu_res_only);
    ctx.sync();
    recordCheck(status,
                vecsNear(gpu_res_only, cpu_res),
                "CUDA time residual-only assembly");
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
  results += femx::tests::cudaAssemblyMatchesCpuReference();
  results += femx::tests::cudaBoundaryMatchesCpuReference();
  results += femx::tests::cudaTimeAssemblyMatchesCpuReference();
  return results.summary();
}
