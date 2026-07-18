#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/linalg/CsrTranspose.hpp>

namespace femx
{
namespace tests
{
namespace
{

bool near(const HostVector& lhs,
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

bool near(const HostCsrMatrix& lhs,
          const HostCsrMatrix& rhs,
          Real                 tolerance = 1.0e-12)
{
  if (lhs.graph().layoutId() != rhs.graph().layoutId())
  {
    return false;
  }
  for (Index k = 0; k < lhs.nnz(); ++k)
  {
    if (std::abs(lhs.valsData()[k] - rhs.valsData()[k]) > tolerance)
    {
      return false;
    }
  }
  return true;
}

void record(TestStatus& status, bool condition, const char* label)
{
  if (!condition)
  {
    std::cout << "    failed check: " << label << '\n';
  }
  status *= condition;
}

TestOutcome persistentCudaCsrOps()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    const HostCsrGraph graph{
        3,
        4,
        HostIndexVector{0, 2, 4, 7},
        HostIndexVector{0, 2, 1, 3, 0, 2, 3}};
    HostCsrMatrix host_mat(graph);
    host_mat.vals() = {2.0, -1.0, 3.0, 4.0, -2.0, 5.0, 1.0};

    const HostCsrTransposeMap host_tr_map(graph);
    HostCsrMatrix             expected_tr(host_tr_map.trGraph());
    trVals(host_mat, host_tr_map, expected_tr);

    const HostVector host_input{1.0, 2.0, 3.0, 4.0};
    const HostVector host_affine_input{1.0, 2.0, 3.0};
    const HostVector host_tr_input{2.0, -1.0, 0.5};

    CudaContext    ctx;
    DeviceCsrGraph device_graph;
    copy(graph, device_graph, ctx);
    DeviceCsrMatrix device_mat(device_graph);
    copy(host_mat, device_mat, ctx);

    DeviceCsrTransposeMap device_tr_map;
    copy(host_tr_map, device_graph, device_tr_map, ctx);
    DeviceCsrMatrix device_tr(device_tr_map.trGraph());
    trVals(device_mat, device_tr_map, device_tr, ctx);

    record(status,
           device_tr_map.srcGraph().rowPtrData()
               == device_graph.rowPtrData(),
           "transpose map retains the existing source graph");

    const Real*  mat_vals = device_mat.valsData();
    const Index* mat_rows = device_mat.rowPtrData();
    const Index* mat_cols = device_mat.colIndData();
    const Real*  tr_vals  = device_tr.valsData();
    const Index* tr_rows  = device_tr.rowPtrData();
    const Index* tr_cols  = device_tr.colIndData();
    const Index* tr_perm  = device_tr_map.srcToTr().data();

    DeviceVector input;
    copy(host_input, input, ctx);
    DeviceVector sliced_input(9);
    copy(input.view(), sliced_input.view().subview(3, 4), ctx);

    DeviceVector output(3);
    apply(device_mat,
          sliced_input.view().subview(3, 4),
          output.view(),
          ctx);
    HostVector first_product;
    copy(output, first_product, ctx);

    DeviceVector copied_product(7);
    copy(output.view(), copied_product.view().subview(2, 3), ctx);
    HostVector copied_storage;
    copy(copied_product, copied_storage, ctx);

    DeviceVector affine_input;
    copy(host_affine_input, affine_input, ctx);
    axpby(-2.0, affine_input.view(), 0.5, output.view(), ctx);
    HostVector affine_result;
    copy(output, affine_result, ctx);

    DeviceVector tr_input;
    copy(host_tr_input, tr_input, ctx);
    DeviceVector tr_product(4);
    apply(device_tr, tr_input.view(), tr_product.view(), ctx);
    HostVector actual_tr_product;
    copy(tr_product, actual_tr_product, ctx);

    HostCsrMatrix actual_tr(host_tr_map.trGraph());
    copy(device_tr, actual_tr, ctx);
    ctx.synchronize();

    record(status,
           near(first_product, HostVector{-1.0, 22.0, 17.0}),
           "rectangular CSR apply");
    record(status,
           near(HostVector(copied_storage.view().subview(2, 3)),
                first_product),
           "device slice copy");
    record(status,
           near(affine_result, HostVector{-2.5, 7.0, 2.5}),
           "device axpby");
    record(status,
           near(actual_tr_product, HostVector{3.0, -3.0, 0.5, -3.5}),
           "transposed CSR apply");
    record(status, near(actual_tr, expected_tr), "transpose value update");

    host_mat.vals() = {-1.0, 2.0, 0.5, -3.0, 4.0, 1.0, -2.0};
    trVals(host_mat, host_tr_map, expected_tr);
    copy(host_mat, device_mat, ctx);
    trVals(device_mat, device_tr_map, device_tr, ctx);
    apply(device_mat,
          sliced_input.view().subview(3, 4),
          output.view(),
          ctx);
    apply(device_tr, tr_input.view(), tr_product.view(), ctx);

    HostVector    updated_product;
    HostVector    updated_tr_product;
    HostCsrMatrix updated_tr(host_tr_map.trGraph());
    copy(output, updated_product, ctx);
    copy(tr_product, updated_tr_product, ctx);
    copy(device_tr, updated_tr, ctx);
    ctx.synchronize();

    record(status,
           near(updated_product, HostVector{5.0, -11.0, -1.0}),
           "updated CSR values");
    record(status,
           near(updated_tr_product, HostVector{0.0, -0.5, 4.5, 2.0}),
           "updated transpose values");
    record(status,
           near(updated_tr, expected_tr),
           "repeated transpose value update");
    record(status,
           mat_vals == device_mat.valsData()
               && mat_rows == device_mat.rowPtrData()
               && mat_cols == device_mat.colIndData()
               && tr_vals == device_tr.valsData()
               && tr_rows == device_tr.rowPtrData()
               && tr_cols == device_tr.colIndData()
               && tr_perm == device_tr_map.srcToTr().data(),
           "operations preserve all persistent allocations");

    bool overlap_rejected = false;
    try
    {
      apply(device_mat,
            sliced_input.view().subview(3, 4),
            sliced_input.view().subview(4, 3),
            ctx);
    }
    catch (const std::runtime_error&)
    {
      overlap_rejected = true;
    }
    record(status, overlap_rejected, "CSR apply rejects overlapping views");
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
  results += femx::tests::persistentCudaCsrOps();
  return results.summary();
}
