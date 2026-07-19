#include <cmath>
#include <exception>
#include <iostream>
#include <utility>

#include "TestHelper.hpp"
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

namespace femx
{
namespace tests
{
namespace
{

struct AffineRowOperator
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

struct TimeRowOperator
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

bool vecsNear(const HostVector& lhs,
              const HostVector& rhs,
              Real              tolerance = 1.0e-12)
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

HostCsrGraph denseThreeByThreeGraph()
{
  return {3,
          3,
          HostIndexVector{0, 3, 6, 9},
          HostIndexVector{0, 1, 2, 0, 1, 2, 0, 1, 2}};
}

void setDenseVals(HostCsrMatrix& mat)
{
  mat.vals() = {4.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0};
}

TestOutcome cudaAssemblyMatchesCpuReference()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
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
    const HostVector host_state{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    HostVector    cpu_res;
    HostCsrMatrix cpu_jac(host_map.graph());
    CpuContext    cpu_ctx;
    assembly::assemble(AffineRowOperator{},
                       hgeom,
                       host_map,
                       host_state,
                       cpu_res,
                       cpu_jac,
                       cpu_ctx);

    CudaContext                                cuda_ctx;
    fem::DeviceGeometry                        dgeom;
    assembly::AssemblyMap<MemorySpace::Device> device_map;
    DeviceVector                               device_state;

    fem::copy(hgeom, dgeom, cuda_ctx);
    assembly::copy(host_map, device_map, cuda_ctx);
    femx::copy(host_state, device_state, cuda_ctx);
    DeviceVector state_clone;
    femx::copy(device_state, state_clone, cuda_ctx);

    DeviceVector    device_res;
    DeviceCsrMatrix device_jac(device_map.graph());
    auto            moved_device_map = std::move(device_map);
    assembly::assemble(AffineRowOperator{},
                       dgeom,
                       moved_device_map,
                       state_clone,
                       device_res,
                       device_jac,
                       cuda_ctx);

    HostVector    gpu_res;
    HostCsrMatrix gpu_jac(host_map.graph());
    femx::copy(device_res, gpu_res, cuda_ctx);
    femx::copy(device_jac, gpu_jac, cuda_ctx);
    cuda_ctx.synchronize();

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
      assembly::assemble(AffineRowOperator{},
                         dgeom,
                         moved_device_map,
                         state_clone,
                         device_jac.vals(),
                         device_jac,
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
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    const HostCsrGraph host_graph = denseThreeByThreeGraph();
    const auto         host_map =
        assembly::makeBoundaryMap(Array<Index>{0, 2}, host_graph);

    HostCsrMatrix expected_mat(host_graph);
    setDenseVals(expected_mat);
    const HostVector initial_rhs{10.0, 20.0, 30.0};
    HostVector       expected_rhs = initial_rhs;
    const HostVector bc_vals{2.0, -1.0};
    assembly::prepareForwardSolve(
        host_map, expected_mat, expected_rhs, bc_vals);

    CudaContext                 ctx;
    DeviceCsrGraph              device_graph;
    assembly::DeviceBoundaryMap device_map;
    femx::copy(host_graph, device_graph, ctx);
    assembly::copy(host_map, device_map, ctx);

    HostCsrMatrix host_mat(host_graph);
    setDenseVals(host_mat);
    DeviceCsrMatrix device_mat(device_graph);
    DeviceVector    device_rhs;
    DeviceVector    device_bc;
    femx::copy(host_mat, device_mat, ctx);
    femx::copy(initial_rhs, device_rhs, ctx);
    femx::copy(bc_vals, device_bc, ctx);

    assembly::prepareForwardSolve(device_map,
                                  device_mat,
                                  device_rhs,
                                  device_bc,
                                  ctx);

    HostCsrMatrix actual_mat(host_graph);
    HostVector    actual_rhs;
    femx::copy(device_mat, actual_mat, ctx);
    femx::copy(device_rhs, actual_rhs, ctx);
    ctx.synchronize();
    recordCheck(status,
                matsNear(actual_mat, expected_mat),
                "CUDA forward mat matches CPU");
    recordCheck(status,
                vecsNear(actual_rhs, expected_rhs),
                "CUDA forward RHS matches CPU");

    HostCsrMatrix expected_hist(host_graph);
    setDenseVals(expected_hist);
    assembly::replaceRows(host_map, expected_hist, 0.0);
    femx::copy(host_mat, device_mat, ctx);
    assembly::replaceRows(device_map, device_mat, 0.0, ctx);
    femx::copy(device_mat, actual_mat, ctx);
    ctx.synchronize();
    recordCheck(status,
                matsNear(actual_mat, expected_hist),
                "CUDA history rows match CPU");

    const HostVector host_state{4.0, 5.0, 6.0};
    const HostVector host_res{10.0, 20.0, 30.0};
    DeviceVector     device_state;
    DeviceVector     device_res;
    femx::copy(host_state, device_state, ctx);
    femx::copy(host_res, device_res, ctx);
    assembly::replaceRes(device_map,
                         device_state.view(),
                         device_bc.view(),
                         device_res.view(),
                         ctx);
    HostVector actual_res;
    femx::copy(device_res, actual_res, ctx);
    ctx.synchronize();
    recordCheck(status,
                vecsNear(actual_res,
                         HostVector{2.0, 20.0, 7.0}),
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

    const HostCsrGraph different_layout{
        3,
        3,
        HostIndexVector{0, 3, 6, 9},
        HostIndexVector{1, 0, 2, 0, 2, 1, 2, 1, 0}};
    DeviceCsrGraph different_device_graph;
    femx::copy(different_layout, different_device_graph, ctx);
    DeviceCsrMatrix wrong_mat(different_device_graph);
    bool            layout_rejected = false;
    try
    {
      femx::copy(host_mat, wrong_mat, ctx);
    }
    catch (const std::runtime_error&)
    {
      layout_rejected = true;
    }
    recordCheck(status,
                layout_rejected,
                "mat copy rejects a different layout");

    const HostCsrGraph diagonal_graph{
        3,
        3,
        HostIndexVector{0, 1, 2, 3},
        HostIndexVector{0, 1, 2}};
    const auto diagonal_map =
        assembly::makeBoundaryMap(Array<Index>{0}, diagonal_graph);
    DeviceCsrGraph              diagonal_device_graph;
    assembly::DeviceBoundaryMap diagonal_device_map;
    femx::copy(diagonal_graph, diagonal_device_graph, ctx);
    assembly::copy(diagonal_map, diagonal_device_map, ctx);
    DeviceCsrMatrix  diag_mat(diagonal_device_graph);
    DeviceVector     diagonal_prescribed;
    const HostVector host_diagonal_prescribed{1.0};
    femx::copy(host_diagonal_prescribed,
               diagonal_prescribed,
               ctx);
    ctx.synchronize();

    bool mat_alias_rejected = false;
    try
    {
      assembly::prepareForwardSolve(diagonal_device_map,
                                    diag_mat,
                                    diag_mat.vals(),
                                    diagonal_prescribed,
                                    ctx);
    }
    catch (const std::runtime_error&)
    {
      mat_alias_rejected = true;
    }
    recordCheck(status,
                mat_alias_rejected,
                "forward preparation rejects mat-val alias");
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
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    const auto map = assembly::makeAssemblyMap(
        3,
        3,
        Array<Array<Index>>{{0, 1}, {1, 2}},
        Array<Array<Index>>{{0, 1}, {1, 2}});
    const HostVector hist{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const HostVector nxt{7.0, 8.0, 9.0};
    HostVector       cpu_res;
    HostCsrMatrix    cpu_jac(map.graph());
    CpuContext       cpu_ctx;
    assembly::assemble(TimeRowOperator{},
                       3,
                       2,
                       state::VariableBlock::hist(1),
                       map,
                       hist,
                       nxt,
                       cpu_res,
                       cpu_jac,
                       cpu_ctx);

    CudaContext                 ctx;
    assembly::DeviceAssemblyMap dmap;
    DeviceVector                dhist;
    DeviceVector                dnxt;
    DeviceVector                dres;
    assembly::copy(map, dmap, ctx);
    femx::copy(hist, dhist, ctx);
    femx::copy(nxt, dnxt, ctx);
    DeviceCsrMatrix djac(dmap.graph());
    assembly::assemble(TimeRowOperator{},
                       3,
                       2,
                       state::VariableBlock::hist(1),
                       dmap,
                       dhist.view(),
                       dnxt.view(),
                       dres,
                       djac,
                       ctx);

    HostVector    gpu_res;
    HostCsrMatrix gpu_jac(map.graph());
    femx::copy(dres, gpu_res, ctx);
    femx::copy(djac, gpu_jac, ctx);
    ctx.synchronize();
    recordCheck(status, vecsNear(gpu_res, cpu_res), "CUDA time res");
    recordCheck(status, matsNear(gpu_jac, cpu_jac), "CUDA time jac");

    assembly::assembleResidual(TimeRowOperator{},
                               3,
                               2,
                               dmap,
                               dhist.view(),
                               dnxt.view(),
                               dres,
                               ctx);
    HostVector gpu_res_only;
    femx::copy(dres, gpu_res_only, ctx);
    ctx.synchronize();
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
