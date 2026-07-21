#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

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

bool equal(const HostIndexVector& lhs, const HostIndexVector& rhs)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }
  for (Index i = 0; i < lhs.size(); ++i)
  {
    if (lhs[i] != rhs[i])
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
    const HostCsrPattern pattern{
        3,
        4,
        HostIndexVector{0, 2, 4, 7},
        HostIndexVector{0, 2, 1, 3, 0, 2, 3}};
    HostCsrMatrix host_mat(pattern);
    host_mat.vals() = {2.0, -1.0, 3.0, 4.0, -2.0, 5.0, 1.0};

    const HostVector host_input{1.0, 2.0, 3.0, 4.0};
    const HostVector host_affine_input{1.0, 2.0, 3.0};
    const HostVector host_tr_input{2.0, -1.0, 0.5};

    CpuContext                cpu_ctx;
    CudaContext               ctx;
    linalg::HostMatrixHandler host_mat_handler(cpu_ctx);
    linalg::CudaVectorHandler vec_handler(ctx);
    linalg::CudaMatrixHandler mat_handler(ctx);
    DeviceCsrPattern          device_graph;
    copy(pattern, device_graph, ctx);
    record(status,
           device_graph.layoutId() == pattern.layoutId(),
           "Device pattern preserves its Host layout identity");
    DeviceCsrMatrix device_mat(device_graph);
    mat_handler.copy(host_mat, device_mat);

    DeviceCsrMatrix device_transpose;
    mat_handler.transpose(device_mat, device_transpose);
    const Index* transpose_row_ptr = device_transpose.rowPtrData();
    const Index* transpose_col_ind = device_transpose.colIndData();
    Real*        transpose_vals    = device_transpose.valsData();

    HostIndexVector actual_transpose_row_ptr;
    HostIndexVector actual_transpose_col_ind;
    HostVector      actual_transpose_vals;
    vec_handler.copy(device_transpose.pattern().rowPtr(),
                     actual_transpose_row_ptr);
    vec_handler.copy(device_transpose.pattern().colInd(),
                     actual_transpose_col_ind);
    vec_handler.copy(device_transpose.vals(), actual_transpose_vals);
    ctx.sync();
    record(status,
           device_transpose.rows() == 4 && device_transpose.cols() == 3
               && device_transpose.nnz() == 7,
           "CUDA CSR transpose dimensions");
    record(status,
           device_transpose.pattern().layoutId() != device_graph.layoutId(),
           "CUDA CSR transpose has a distinct layout identity");
    record(status,
           equal(actual_transpose_row_ptr,
                 HostIndexVector{0, 2, 3, 5, 7}),
           "CUDA CSR transpose row offsets");
    record(status,
           equal(actual_transpose_col_ind,
                 HostIndexVector{0, 2, 1, 0, 2, 1, 2}),
           "CUDA CSR transpose column indices");
    record(status,
           near(actual_transpose_vals,
                HostVector{2.0, -2.0, 3.0, -1.0, 5.0, 4.0, 1.0}),
           "CUDA CSR transpose values");

    const Real*  mat_vals = device_mat.valsData();
    const Index* mat_rows = device_mat.rowPtrData();
    const Index* mat_cols = device_mat.colIndData();

    DeviceVector input;
    vec_handler.copy(host_input, input);
    DeviceVector squared_norm(1);
    vec_handler.squaredNorm(input.view(), squared_norm.view());
    HostVector actual_squared_norm;
    vec_handler.copy(squared_norm, actual_squared_norm);
    DeviceVector sliced_input(9);
    vec_handler.copy(input.view(), sliced_input.view().subview(3, 4));

    DeviceVector output(3);
    mat_handler.matvec(device_mat,
                       sliced_input.view().subview(3, 4),
                       output.view());
    HostVector first_product;
    vec_handler.copy(output, first_product);

    DeviceVector copied_product(7);
    vec_handler.copy(output.view(), copied_product.view().subview(2, 3));
    HostVector copied_storage;
    vec_handler.copy(copied_product, copied_storage);

    DeviceVector affine_input;
    vec_handler.copy(host_affine_input, affine_input);
    vec_handler.axpby(-2.0, affine_input.view(), 0.5, output.view());
    HostVector affine_result;
    vec_handler.copy(output, affine_result);

    DeviceVector tr_input;
    vec_handler.copy(host_tr_input, tr_input);
    DeviceVector direct_tr_product(4);
    mat_handler.matvecT(device_mat,
                        tr_input.view(),
                        direct_tr_product.view());
    HostVector actual_direct_tr_product;
    vec_handler.copy(direct_tr_product, actual_direct_tr_product);
    HostVector expected_tr_product(4);
    host_mat_handler.matvecT(host_mat,
                             host_tr_input.view(),
                             expected_tr_product.view());
    ctx.sync();

    record(status,
           near(first_product, HostVector{-1.0, 22.0, 17.0}),
           "rectangular CSR apply");
    record(status,
           std::abs(actual_squared_norm[0] - 30.0) <= 1.0e-12,
           "cuBLAS squared norm");
    record(status,
           near(HostVector(copied_storage.view().subview(2, 3)),
                first_product),
           "device slice copy");
    record(status,
           near(affine_result, HostVector{-2.5, 7.0, 2.5}),
           "device axpby");

    const HostIndexVector host_indices{3, 0, 2};
    DeviceIndexVector     indices;
    vec_handler.copy(host_indices, indices);
    DeviceVector gathered(3);
    vec_handler.gather(input.view(), indices.view(), gathered.view());
    DeviceVector scattered(4);
    vec_handler.zero(scattered.view());
    vec_handler.scatter(gathered.view(), indices.view(), scattered.view());
    HostVector actual_gathered;
    HostVector actual_scattered;
    vec_handler.copy(gathered, actual_gathered);
    vec_handler.copy(scattered, actual_scattered);
    ctx.sync();
    record(status,
           near(actual_gathered, HostVector{4.0, 1.0, 3.0}),
           "cuSPARSE gather");
    record(status,
           near(actual_scattered, HostVector{1.0, 0.0, 3.0, 4.0}),
           "cuSPARSE scatter");

    const HostVector host_dense{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    DeviceVector     device_dense;
    vec_handler.copy(host_dense, device_dense);
    DeviceVector dense_product(2);
    mat_handler.matvec(DeviceMatrixView<const Real>(device_dense.data(), 2, 3),
                       input.view().subview(0, 3),
                       dense_product.view());
    DeviceVector dense_tr_product(3);
    mat_handler.matvecT(DeviceMatrixView<const Real>(device_dense.data(), 2, 3),
                        dense_product.view(),
                        dense_tr_product.view());
    HostVector actual_dense;
    HostVector actual_dense_tr;
    vec_handler.copy(dense_product, actual_dense);
    vec_handler.copy(dense_tr_product, actual_dense_tr);
    ctx.sync();
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
    mat_handler.copy(host_mat, device_mat);
    mat_handler.transpose(device_mat, device_transpose);
    mat_handler.matvec(device_mat,
                       sliced_input.view().subview(3, 4),
                       output.view());
    mat_handler.matvecT(device_mat,
                        tr_input.view(),
                        direct_tr_product.view());

    HostVector updated_product;
    HostVector updated_direct_tr_product;
    HostVector updated_transpose_vals;
    vec_handler.copy(output, updated_product);
    vec_handler.copy(direct_tr_product, updated_direct_tr_product);
    vec_handler.copy(device_transpose.vals(), updated_transpose_vals);
    host_mat_handler.matvecT(host_mat,
                             host_tr_input.view(),
                             expected_tr_product.view());
    ctx.sync();

    record(status,
           near(updated_product, HostVector{5.0, -11.0, -1.0}),
           "updated CSR values");
    record(status,
           near(updated_direct_tr_product, expected_tr_product),
           "updated transpose values");
    record(status,
           near(updated_transpose_vals,
                HostVector{-1.0, 4.0, 0.5, 2.0, 1.0, -3.0, -2.0}),
           "updated explicit transpose values");
    record(status,
           mat_vals == device_mat.valsData()
               && mat_rows == device_mat.rowPtrData()
               && mat_cols == device_mat.colIndData()
               && transpose_row_ptr == device_transpose.rowPtrData()
               && transpose_col_ind == device_transpose.colIndData()
               && transpose_vals == device_transpose.valsData(),
           "operations preserve all persistent allocations");

    bool overlap_rejected = false;
    try
    {
      mat_handler.matvec(device_mat,
                         sliced_input.view().subview(3, 4),
                         sliced_input.view().subview(4, 3));
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
