#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Dense.hpp>

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

    const HostVector host_input{1.0, 2.0, 3.0, 4.0};
    const HostVector host_affine_input{1.0, 2.0, 3.0};
    const HostVector host_tr_input{2.0, -1.0, 0.5};

    CpuContext     cpu_ctx;
    CudaContext    ctx;
    DeviceCsrGraph device_graph;
    copy(graph, device_graph, ctx);
    DeviceCsrMatrix device_mat(device_graph);
    copy(host_mat, device_mat, ctx);

    const Real*  mat_vals = device_mat.valsData();
    const Index* mat_rows = device_mat.rowPtrData();
    const Index* mat_cols = device_mat.colIndData();

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
    DeviceVector direct_tr_product(4);
    applyT(device_mat,
           tr_input.view(),
           direct_tr_product.view(),
           ctx);
    HostVector actual_direct_tr_product;
    copy(direct_tr_product, actual_direct_tr_product, ctx);
    HostVector expected_tr_product(4);
    applyT(host_mat,
           host_tr_input.view(),
           expected_tr_product.view(),
           cpu_ctx);
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

    const HostIndexVector host_indices{3, 0, 2};
    DeviceIndexVector     indices;
    copy(host_indices, indices, ctx);
    DeviceVector gathered(3);
    gather(input.view(), indices.view(), gathered.view(), ctx);
    DeviceVector scattered(4);
    scattered.setZero(ctx);
    scatter(gathered.view(), indices.view(), scattered.view(), ctx);
    HostVector actual_gathered;
    HostVector actual_scattered;
    copy(gathered, actual_gathered, ctx);
    copy(scattered, actual_scattered, ctx);
    ctx.synchronize();
    record(status,
           near(actual_gathered, HostVector{4.0, 1.0, 3.0}),
           "cuSPARSE gather");
    record(status,
           near(actual_scattered, HostVector{1.0, 0.0, 3.0, 4.0}),
           "cuSPARSE scatter");

    const HostVector host_dense{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    DeviceVector     device_dense;
    copy(host_dense, device_dense, ctx);
    DeviceVector dense_product(2);
    femx::apply(DeviceMatrixView<const Real>(device_dense.data(), 2, 3),
                input.view().subview(0, 3),
                dense_product.view(),
                ctx);
    DeviceVector dense_tr_product(3);
    femx::applyT(DeviceMatrixView<const Real>(device_dense.data(), 2, 3),
                 dense_product.view(),
                 dense_tr_product.view(),
                 ctx);
    HostVector actual_dense;
    HostVector actual_dense_tr;
    copy(dense_product, actual_dense, ctx);
    copy(dense_tr_product, actual_dense_tr, ctx);
    ctx.synchronize();
    record(status,
           near(actual_dense, HostVector{14.0, 32.0}),
           "cuBLAS row-major dense apply");
    record(status,
           near(actual_dense_tr, HostVector{142.0, 188.0, 234.0}),
           "cuBLAS row-major dense transpose");

    record(status,
           near(actual_direct_tr_product, expected_tr_product),
           "transposed CSR apply");

    host_mat.vals() = {-1.0, 2.0, 0.5, -3.0, 4.0, 1.0, -2.0};
    copy(host_mat, device_mat, ctx);
    apply(device_mat,
          sliced_input.view().subview(3, 4),
          output.view(),
          ctx);
    applyT(device_mat,
           tr_input.view(),
           direct_tr_product.view(),
           ctx);

    HostVector updated_product;
    HostVector updated_direct_tr_product;
    copy(output, updated_product, ctx);
    copy(direct_tr_product, updated_direct_tr_product, ctx);
    applyT(host_mat,
           host_tr_input.view(),
           expected_tr_product.view(),
           cpu_ctx);
    ctx.synchronize();

    record(status,
           near(updated_product, HostVector{5.0, -11.0, -1.0}),
           "updated CSR values");
    record(status,
           near(updated_direct_tr_product, expected_tr_product),
           "updated transpose values");
    record(status,
           mat_vals == device_mat.valsData()
               && mat_rows == device_mat.rowPtrData()
               && mat_cols == device_mat.colIndData(),
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
