#include <cmath>
#include <exception>
#include <iostream>
#include <utility>

#include <TestHelper.hpp>
#include <femx/assembly/Assembly.hpp>
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaJacobian.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/native/HostJacobian.hpp>

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
  ctx.vectors().copy(source.vals().view(), destination.vals().view());
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
  ctx.vectors().copy(source.vals().view(), destination.vals().view());
}

void loadMatrix(const HostCsrMatrix&  source,
                linalg::HostJacobian& destination)
{
  destination.begin(source.pattern());
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

void loadMatrix(const HostCsrMatrix&  source,
                linalg::CudaJacobian& destination,
                linalg::CudaContext&  ctx)
{
  destination.begin(source.pattern());
  ctx.vectors().copy(source.vals().view(),
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

    const fem::HostGeometry hgeom = fem::makeGeometry(mesh);
    const auto              host_map =
        assembly::makeAssemblyMap(fem::DofLayout(space));
    const HostVector<Real> host_state{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    HostVector<Real>     cpu_res;
    linalg::HostContext  cpu_ctx;
    linalg::HostJacobian cpu_jacobian(cpu_ctx);
    cpu_jacobian.begin(host_map.pattern());
    assembly::assemble(AffineElementKernel{},
                       hgeom,
                       host_map,
                       host_state,
                       cpu_res,
                       cpu_jacobian,
                       cpu_ctx);
    const HostCsrMatrix& cpu_jac = cpu_jacobian.matrix();

    linalg::CudaContext         cuda_ctx;
    auto&                       vec_handler = cuda_ctx.vectors();
    fem::DeviceGeometry         dgeom;
    assembly::DeviceAssemblyMap device_map;
    DeviceVector<Real>          device_state;

    fem::copy(hgeom, dgeom, cuda_ctx);
    assembly::copy(host_map, device_map, cuda_ctx);
    vec_handler.copy(host_state, device_state);
    DeviceVector<Real> state_clone;
    vec_handler.copy(device_state, state_clone);

    DeviceVector<Real>   device_res;
    linalg::CudaJacobian device_jacobian(cuda_ctx);
    device_jacobian.begin(host_map.pattern());
    auto moved_device_map = std::move(device_map);
    assembly::assemble(AffineElementKernel{},
                       dgeom,
                       moved_device_map,
                       state_clone,
                       device_res,
                       device_jacobian,
                       cuda_ctx);

    HostVector<Real> gpu_res;
    HostCsrMatrix    gpu_jac(host_map.pattern());
    vec_handler.copy(device_res, gpu_res);
    copyMatrix(device_jacobian.matrix(), gpu_jac, cuda_ctx);
    cuda_ctx.sync();

    recordCheck(status,
                vecsNear(gpu_res, cpu_res),
                "CUDA res matches CPU");
    recordCheck(status,
                matsNear(gpu_jac, cpu_jac),
                "CUDA Jacobian matches CPU");
    recordCheck(status,
                hgeom.maxElemNodes() == 4,
                "geometry maximum element nodes");

    bool mat_alias_rejected = false;
    try
    {
      assembly::assemble(AffineElementKernel{},
                         dgeom,
                         moved_device_map,
                         state_clone,
                         state_clone,
                         device_jacobian,
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
    const HostCsrPattern host_graph = denseThreeByThreeGraph();
    const auto           host_map =
        assembly::makeBoundaryMap(HostVector<Index>{0, 2});

    HostCsrMatrix host_mat(host_graph);
    setDenseVals(host_mat);
    const HostVector<Real> initial_rhs{10.0, 20.0, 30.0};
    HostVector<Real>       expected_rhs = initial_rhs;
    const HostVector<Real> bc_vals{2.0, -1.0};
    linalg::HostContext    host_ctx;
    linalg::HostJacobian   expected_jacobian(host_ctx);
    loadMatrix(host_mat, expected_jacobian);
    expected_jacobian.eliminateColumns(host_map.view().constrained_rows,
                                       bc_vals.view(),
                                       expected_rhs.view());
    const HostCsrMatrix& expected_mat =
        expected_jacobian.matrix();

    linalg::CudaContext         ctx;
    auto&                       vec_handler = ctx.vectors();
    assembly::DeviceBoundaryMap device_map;
    assembly::copy(host_map, device_map, ctx);

    linalg::CudaJacobian device_jacobian(ctx);
    loadMatrix(host_mat, device_jacobian, ctx);
    DeviceVector<Real> device_rhs;
    DeviceVector<Real> device_bc;
    vec_handler.copy(initial_rhs, device_rhs);
    vec_handler.copy(bc_vals, device_bc);

    device_jacobian.eliminateColumns(
        device_map.view().constrained_rows,
        device_bc.view(),
        device_rhs.view());

    HostCsrMatrix    actual_mat(host_graph);
    HostVector<Real> actual_rhs;
    copyMatrix(device_jacobian.matrix(), actual_mat, ctx);
    vec_handler.copy(device_rhs, actual_rhs);
    ctx.sync();
    recordCheck(status,
                matsNear(actual_mat, expected_mat),
                "CUDA forward mat matches CPU");
    recordCheck(status,
                vecsNear(actual_rhs, expected_rhs),
                "CUDA forward RHS matches CPU");

    linalg::HostJacobian expected_hist_jacobian(host_ctx);
    loadMatrix(host_mat, expected_hist_jacobian);
    expected_hist_jacobian.replaceRows(
        host_map.view().constrained_rows, 0.0);
    const HostCsrMatrix& expected_hist =
        expected_hist_jacobian.matrix();

    loadMatrix(host_mat, device_jacobian, ctx);
    device_jacobian.replaceRows(
        device_map.view().constrained_rows, 0.0);
    copyMatrix(device_jacobian.matrix(), actual_mat, ctx);
    ctx.sync();
    recordCheck(status,
                matsNear(actual_mat, expected_hist),
                "CUDA history rows match CPU");

    const HostVector<Real> host_state{4.0, 5.0, 6.0};
    const HostVector<Real> host_res{10.0, 20.0, 30.0};
    DeviceVector<Real>     device_state;
    DeviceVector<Real>     device_res;
    vec_handler.copy(host_state, device_state);
    vec_handler.copy(host_res, device_res);
    assembly::replaceRes(device_map,
                         device_state.view(),
                         device_bc.view(),
                         device_res.view(),
                         ctx);
    HostVector<Real> actual_res;
    vec_handler.copy(device_res, actual_res);
    ctx.sync();
    recordCheck(status,
                vecsNear(actual_res,
                         HostVector<Real>{2.0, 20.0, 7.0}),
                "CUDA res replacement");

    bool alias_rejected = false;
    try
    {
      assembly::replaceRes(device_map,
                           device_state.view(),
                           device_bc.view(),
                           device_state.view(),
                           ctx);
    }
    catch (const std::runtime_error&)
    {
      alias_rejected = true;
    }
    recordCheck(status,
                alias_rejected,
                "res replacement rejects output alias");

    const HostCsrPattern different_layout{
        3,
        3,
        HostVector<Index>{0, 3, 6, 9},
        HostVector<Index>{1, 0, 2, 0, 2, 1, 2, 1, 0}};
    DeviceCsrPattern different_device_graph;
    femx::copy(different_layout, different_device_graph, ctx);
    DeviceCsrMatrix wrong_mat(different_device_graph);
    bool            layout_rejected = false;
    try
    {
      copyMatrix(host_mat, wrong_mat, ctx);
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
    const HostVector<Real> hist{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const HostVector<Real> nxt{7.0, 8.0, 9.0};
    HostVector<Real>       cpu_res;
    linalg::HostContext    cpu_ctx;
    linalg::HostJacobian   cpu_jacobian(cpu_ctx);
    cpu_jacobian.begin(map.pattern());
    assembly::assemble(TimeElementKernel{},
                       3,
                       2,
                       state::VariableBlock::hist(1),
                       map,
                       0,
                       map.numElems(),
                       hist.view(),
                       nxt.view(),
                       cpu_res,
                       cpu_jacobian,
                       cpu_ctx);
    const HostCsrMatrix& cpu_jac = cpu_jacobian.matrix();

    linalg::CudaContext         ctx;
    auto&                       vec_handler = ctx.vectors();
    assembly::DeviceAssemblyMap dmap;
    DeviceVector<Real>          dhist;
    DeviceVector<Real>          dnxt;
    DeviceVector<Real>          dres;
    assembly::copy(map, dmap, ctx);
    vec_handler.copy(hist, dhist);
    vec_handler.copy(nxt, dnxt);
    linalg::CudaJacobian djac(ctx);
    djac.begin(map.pattern());
    assembly::assemble(TimeElementKernel{},
                       3,
                       2,
                       state::VariableBlock::hist(1),
                       dmap,
                       dhist.view(),
                       dnxt.view(),
                       dres,
                       djac,
                       ctx);

    HostVector<Real> gpu_res;
    HostCsrMatrix    gpu_jac(map.pattern());
    vec_handler.copy(dres, gpu_res);
    copyMatrix(djac.matrix(), gpu_jac, ctx);
    ctx.sync();
    recordCheck(status, vecsNear(gpu_res, cpu_res), "CUDA time res");
    recordCheck(status, matsNear(gpu_jac, cpu_jac), "CUDA time jac");

    assembly::assembleResidual(TimeElementKernel{},
                               3,
                               2,
                               dmap,
                               dhist.view(),
                               dnxt.view(),
                               dres,
                               ctx);
    HostVector<Real> gpu_res_only;
    vec_handler.copy(dres, gpu_res_only);
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
