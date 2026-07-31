#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include <TestHelper.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/native/HostSystemMatrix.hpp>

namespace femx
{
namespace tests
{
namespace
{

bool near(const HostVector<Real>& lhs,
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

bool equal(const HostVector<Index>& lhs, const HostVector<Index>& rhs)
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

TestOutcome persistentCudaCsrOps()
{
  TestStatus status(__func__);
  if (!linalg::CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    const HostCsrPattern pattern{
        3,
        4,
        HostVector<Index>{0, 2, 4, 7},
        HostVector<Index>{0, 2, 1, 3, 0, 2, 3}};
    HostCsrMatrix h_mat(pattern);
    h_mat.vals() = {2.0, -1.0, 3.0, 4.0, -2.0, 5.0, 1.0};

    const HostVector<Real> h_input{1.0, 2.0, 3.0, 4.0};
    const HostVector<Real> h_affine_input{1.0, 2.0, 3.0};
    const HostVector<Real> h_trans_input{2.0, -1.0, 0.5};

    linalg::HostContext      cpu_ctx;
    linalg::CudaContext      ctx;
    linalg::HostSystemMatrix h_jac(cpu_ctx);
    linalg::CudaSystemMatrix jac(ctx);
    auto&                    vec_handler = ctx.vectorHandler();

    DeviceVector<Real> assigned_reals;
    vec_handler.assign(assigned_reals, 3, 2);
    HostVector<Real> actual_assigned_reals;
    vec_handler.copy(assigned_reals, actual_assigned_reals);

    DeviceVector<Index> assigned_indices;
    vec_handler.assign(assigned_indices, 3, -1);
    HostVector<Index> actual_assigned_indices;
    vec_handler.copy(assigned_indices, actual_assigned_indices);
    ctx.sync();
    record(status,
           near(actual_assigned_reals, HostVector<Real>{2.0, 2.0, 2.0}),
           "CUDA Real assignment");
    record(status,
           equal(actual_assigned_indices, HostVector<Index>{-1, -1, -1}),
           "CUDA Index assignment");

    DeviceCsrPattern d_graph;
    copy(pattern, d_graph, ctx);
    record(status,
           d_graph.layoutId() == pattern.layoutId(),
           "Device pattern preserves its Host layout identity");
    DeviceCsrMatrix d_mat(d_graph);
    copyMatrix(h_mat, d_mat, ctx);

    DeviceCsrMatrix d_transpose;
    jac.transpose(d_mat, d_transpose);
    const Index* transpose_row_ptr = d_transpose.rowPtrData();
    const Index* transpose_col_ind = d_transpose.colIndData();
    Real*        transpose_vals    = d_transpose.valsData();

    HostVector<Index> actual_transpose_row_ptr;
    HostVector<Index> actual_transpose_col_ind;
    HostVector<Real>  actual_transpose_vals;
    vec_handler.copy(d_transpose.pattern().rowPtr(),
                     actual_transpose_row_ptr);
    vec_handler.copy(d_transpose.pattern().colInd(),
                     actual_transpose_col_ind);
    vec_handler.copy(d_transpose.vals(), actual_transpose_vals);
    ctx.sync();
    record(status,
           d_transpose.rows() == 4 && d_transpose.cols() == 3
               && d_transpose.nnz() == 7,
           "CUDA CSR transpose dimensions");
    record(status,
           d_transpose.pattern().layoutId() != d_graph.layoutId(),
           "CUDA CSR transpose has a distinct layout identity");
    record(status,
           equal(actual_transpose_row_ptr,
                 HostVector<Index>{0, 2, 3, 5, 7}),
           "CUDA CSR transpose row offsets");
    record(status,
           equal(actual_transpose_col_ind,
                 HostVector<Index>{0, 2, 1, 0, 2, 1, 2}),
           "CUDA CSR transpose column indices");
    record(status,
           near(actual_transpose_vals,
                HostVector<Real>{2.0, -2.0, 3.0, -1.0, 5.0, 4.0, 1.0}),
           "CUDA CSR transpose values");

    const Real*  mat_vals = d_mat.valsData();
    const Index* mat_rows = d_mat.rowPtrData();
    const Index* mat_cols = d_mat.colIndData();

    DeviceVector<Real> input;
    vec_handler.copy(h_input, input);
    DeviceVector<Real> squared_norm(1);
    vec_handler.squaredNorm(input.view(), squared_norm.view());
    HostVector<Real> actual_squared_norm;
    vec_handler.copy(squared_norm, actual_squared_norm);
    DeviceVector<Real> sliced_input(9);
    vec_handler.copy(input.view(), sliced_input.view().subview(3, 4));

    DeviceVector<Real> output(3);
    jac.apply(d_mat,
              sliced_input.view().subview(3, 4),
              output.view());
    HostVector<Real> first_product;
    vec_handler.copy(output, first_product);

    DeviceVector<Real> copied_product(7);
    vec_handler.copy(output.view(), copied_product.view().subview(2, 3));
    HostVector<Real> copied_storage;
    vec_handler.copy(copied_product, copied_storage);

    DeviceVector<Real> affine_input;
    vec_handler.copy(h_affine_input, affine_input);
    vec_handler.axpby(-2.0, affine_input.view(), 0.5, output.view());
    HostVector<Real> affine_result;
    vec_handler.copy(output, affine_result);

    DeviceVector<Real> d_trans_input;
    vec_handler.copy(h_trans_input, d_trans_input);
    DeviceVector<Real> d_direct_trans_product(4);
    jac.applyT(d_mat,
               d_trans_input.view(),
               d_direct_trans_product.view());
    HostVector<Real> actual_direct_trans_product;
    vec_handler.copy(d_direct_trans_product, actual_direct_trans_product);
    HostVector<Real> expected_trans_product(4);
    h_jac.applyT(h_mat,
                 h_trans_input.view(),
                 expected_trans_product.view());
    ctx.sync();

    record(status,
           near(first_product, HostVector<Real>{-1.0, 22.0, 17.0}),
           "rectangular CSR apply");
    record(status,
           std::abs(actual_squared_norm[0] - 30.0) <= 1.0e-12,
           "cuBLAS squared norm");
    record(status,
           near(HostVector<Real>(copied_storage.view().subview(2, 3)),
                first_product),
           "device slice copy");
    record(status,
           near(affine_result, HostVector<Real>{-2.5, 7.0, 2.5}),
           "device axpby");

    const HostVector<Index> h_indices{3, 0, 2};
    DeviceVector<Index>     indices;
    vec_handler.copy(h_indices, indices);
    DeviceVector<Real> gathered(3);
    vec_handler.gather(input.view(), indices.view(), gathered.view());
    DeviceVector<Real> scattered(4);
    vec_handler.zero(scattered.view());
    vec_handler.scatter(gathered.view(), indices.view(), scattered.view());
    HostVector<Real> actual_gathered;
    HostVector<Real> actual_scattered;
    vec_handler.copy(gathered, actual_gathered);
    vec_handler.copy(scattered, actual_scattered);
    ctx.sync();
    record(status,
           near(actual_gathered, HostVector<Real>{4.0, 1.0, 3.0}),
           "cuSPARSE gather");
    record(status,
           near(actual_scattered, HostVector<Real>{1.0, 0.0, 3.0, 4.0}),
           "cuSPARSE scatter");

    const HostVector<Real> h_dense{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    DeviceVector<Real>     d_dense;
    vec_handler.copy(h_dense, d_dense);
    DeviceVector<Real> dense_product(2);
    jac.apply(DeviceMatrixView<const Real>(d_dense.data(), 2, 3),
              input.view().subview(0, 3),
              dense_product.view());
    DeviceVector<Real> d_dense_trans_product(3);
    jac.applyT(DeviceMatrixView<const Real>(d_dense.data(), 2, 3),
               dense_product.view(),
               d_dense_trans_product.view());
    HostVector<Real> actual_dense;
    HostVector<Real> actual_dense_trans;
    vec_handler.copy(dense_product, actual_dense);
    vec_handler.copy(d_dense_trans_product, actual_dense_trans);
    ctx.sync();
    record(status,
           near(actual_dense, HostVector<Real>{14.0, 32.0}),
           "cuBLAS row-major dense apply");
    record(status,
           near(actual_dense_trans, HostVector<Real>{142.0, 188.0, 234.0}),
           "cuBLAS row-major dense transpose");

    record(status,
           near(actual_direct_trans_product, expected_trans_product),
           "transposed CSR apply");

    h_mat.vals() = {-1.0, 2.0, 0.5, -3.0, 4.0, 1.0, -2.0};
    copyMatrix(h_mat, d_mat, ctx);
    jac.transpose(d_mat, d_transpose);
    jac.apply(d_mat,
              sliced_input.view().subview(3, 4),
              output.view());
    jac.applyT(d_mat,
               d_trans_input.view(),
               d_direct_trans_product.view());

    HostVector<Real> updated_product;
    HostVector<Real> updated_direct_trans_product;
    HostVector<Real> updated_transpose_vals;
    vec_handler.copy(output, updated_product);
    vec_handler.copy(d_direct_trans_product, updated_direct_trans_product);
    vec_handler.copy(d_transpose.vals(), updated_transpose_vals);
    h_jac.applyT(h_mat,
                 h_trans_input.view(),
                 expected_trans_product.view());
    ctx.sync();

    record(status,
           near(updated_product, HostVector<Real>{5.0, -11.0, -1.0}),
           "updated CSR values");
    record(status,
           near(updated_direct_trans_product, expected_trans_product),
           "updated transpose values");
    record(status,
           near(updated_transpose_vals,
                HostVector<Real>{-1.0, 4.0, 0.5, 2.0, 1.0, -3.0, -2.0}),
           "updated explicit transpose values");
    record(status,
           mat_vals == d_mat.valsData()
               && mat_rows == d_mat.rowPtrData()
               && mat_cols == d_mat.colIndData()
               && transpose_row_ptr == d_transpose.rowPtrData()
               && transpose_col_ind == d_transpose.colIndData()
               && transpose_vals == d_transpose.valsData(),
           "operations preserve all persistent allocations");

    bool overlap_rejected = false;
    try
    {
      jac.apply(d_mat,
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
