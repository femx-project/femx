#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <femx/common/Checks.hpp>
#include <femx/linalg/CsrMatrix.hpp>

namespace femx
{
namespace
{
constexpr int kThreads = 256;

void checkCusparse(cusparseStatus_t status, const char* op)
{
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error(std::string(op) + ": "
                             + cusparseGetErrorString(status));
  }
}

class CusparseHandle
{
public:
  CusparseHandle()
  {
    checkCusparse(cusparseCreate(&handle_), "cusparseCreate failed");
  }

  ~CusparseHandle()
  {
    if (handle_ != nullptr)
    {
      cusparseDestroy(handle_);
    }
  }

  CusparseHandle(const CusparseHandle&)            = delete;
  CusparseHandle& operator=(const CusparseHandle&) = delete;

  cusparseHandle_t get(void* stream)
  {
    checkCusparse(
        cusparseSetStream(handle_, static_cast<cudaStream_t>(stream)),
        "cusparseSetStream failed");
    return handle_;
  }

private:
  cusparseHandle_t handle_{nullptr};
};

cusparseHandle_t cusparseHandle(void* stream)
{
  thread_local CusparseHandle handle;
  return handle.get(stream);
}

struct SpmvOperation
{
  cusparseSpMatDescr_t matrix{nullptr};
  cusparseDnVecDescr_t x{nullptr};
  cusparseDnVecDescr_t y{nullptr};
  void*                workspace{nullptr};
  std::size_t          workspace_capacity{0};
  bool                 preprocessed{false};

  ~SpmvOperation()
  {
    // cudaFree waits for queued uses of the workspace before descriptors are
    // destroyed, including when the owning matrix dies before its context.
    device::release(workspace);
    if (y != nullptr)
    {
      cusparseDestroyDnVec(y);
    }
    if (x != nullptr)
    {
      cusparseDestroyDnVec(x);
    }
    if (matrix != nullptr)
    {
      cusparseDestroySpMat(matrix);
    }
  }

  SpmvOperation()                                = default;
  SpmvOperation(const SpmvOperation&)            = delete;
  SpmvOperation& operator=(const SpmvOperation&) = delete;
};

struct MatrixKey
{
  const Index* row_ptr{nullptr};
  const Index* col_ind{nullptr};
  const Real*  vals{nullptr};
  Index        rows{0};
  Index        cols{0};
  Index        nnz{0};

  friend bool operator==(const MatrixKey& lhs,
                         const MatrixKey& rhs) noexcept
  {
    return lhs.row_ptr == rhs.row_ptr && lhs.col_ind == rhs.col_ind
           && lhs.vals == rhs.vals && lhs.rows == rhs.rows
           && lhs.cols == rhs.cols && lhs.nnz == rhs.nnz;
  }
};

struct MatrixKeyHash
{
  std::size_t operator()(const MatrixKey& key) const noexcept
  {
    std::size_t hash = std::hash<const void*>{}(key.row_ptr);
    const auto  mix  = [&hash](std::size_t val)
    {
      hash ^= val + std::size_t{0x9e3779b9} + (hash << 6) + (hash >> 2);
    };
    mix(std::hash<const void*>{}(key.col_ind));
    mix(std::hash<const void*>{}(key.vals));
    mix(std::hash<Index>{}(key.rows));
    mix(std::hash<Index>{}(key.cols));
    mix(std::hash<Index>{}(key.nnz));
    return hash;
  }
};

struct MatrixSpmvState
{
  SpmvOperation apply;
  SpmvOperation apply_transpose;
};

struct CsrSpmvState
{
  std::mutex mutex;
  std::unordered_map<MatrixKey,
                     std::unique_ptr<MatrixSpmvState>,
                     MatrixKeyHash>
      matrices;
};

template <class T>
void checkView(const T* data, Index size, const char* name)
{
  require(size >= 0 && (size == 0 || data != nullptr), name);
}

unsigned int blocks(Index size)
{
  return static_cast<unsigned int>(
      (static_cast<std::int64_t>(size) + kThreads - 1) / kThreads);
}

__global__ void scaleKernel(Index size, Real scale, Real* vals)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    vals[i] *= scale;
  }
}

void checkApply(const DeviceCsrMatrix& mat,
                DeviceConstVectorView  x,
                DeviceVectorView       y,
                bool                   transpose)
{
  checkView(x.data(), x.size(), "CSR apply has an invalid input view");
  checkView(y.data(), y.size(), "CSR apply has an invalid output view");
  const Index in_size  = transpose ? mat.rows() : mat.cols();
  const Index out_size = transpose ? mat.cols() : mat.rows();
  require(x.size() == in_size && y.size() == out_size,
          "CSR apply vector size mismatch");
  require(mat.rows() == 0 || mat.rowPtrData() != nullptr,
          "CSR apply has no Device row offsets");
  require(mat.nnz() == 0
              || (mat.colIndData() != nullptr && mat.valsData() != nullptr),
          "CSR apply has incomplete Device storage");
  require(!detail::overlaps(x, y),
          "CSR apply does not support in-place vectors");
  require(!detail::overlaps(mat.valsData(), mat.nnz(), y.data(), y.size()),
          "CSR apply output aliases matrix values");
}

void scaleOutput(DeviceVectorView y, Real beta, CudaContext& ctx)
{
  if (y.empty() || beta == 1.0)
  {
    return;
  }
  if (beta == 0.0)
  {
    device::zero(y.data(),
                 static_cast<std::size_t>(y.size()) * sizeof(Real),
                 ctx.stream());
    return;
  }
  scaleKernel<<<blocks(y.size()),
                kThreads,
                0,
                static_cast<cudaStream_t>(ctx.stream())>>>(
      y.size(), beta, y.data());
  device::checkLastError();
}
} // namespace

namespace detail
{
struct CudaContextAccess
{
  static std::shared_ptr<void>& sparseState(CudaContext& ctx)
  {
    return ctx.sparse_state_;
  }
};
} // namespace detail

namespace
{
CsrSpmvState& spmvState(CudaContext& ctx)
{
  auto& storage = detail::CudaContextAccess::sparseState(ctx);
  if (!storage)
  {
    storage = std::shared_ptr<void>(
        new CsrSpmvState,
        [](void* state)
        { delete static_cast<CsrSpmvState*>(state); });
  }
  return *static_cast<CsrSpmvState*>(storage.get());
}

MatrixKey matrixKey(const DeviceCsrMatrix& mat) noexcept
{
  return {mat.rowPtrData(),
          mat.colIndData(),
          mat.valsData(),
          mat.rows(),
          mat.cols(),
          mat.nnz()};
}

void ensureDescriptors(SpmvOperation&         op,
                       const DeviceCsrMatrix& mat,
                       DeviceConstVectorView  x,
                       DeviceVectorView       y)
{
  if (op.matrix == nullptr)
  {
    checkCusparse(
        cusparseCreateCsr(&op.matrix,
                          mat.rows(),
                          mat.cols(),
                          mat.nnz(),
                          const_cast<Index*>(mat.rowPtrData()),
                          const_cast<Index*>(mat.colIndData()),
                          const_cast<Real*>(mat.valsData()),
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_64F),
        "cusparseCreateCsr failed");
  }
  if (op.x == nullptr)
  {
    checkCusparse(cusparseCreateDnVec(
                      &op.x,
                      x.size(),
                      const_cast<Real*>(x.data()),
                      CUDA_R_64F),
                  "cusparseCreateDnVec(input) failed");
  }
  else
  {
    checkCusparse(
        cusparseDnVecSetValues(op.x, const_cast<Real*>(x.data())),
        "cusparseDnVecSetValues(input) failed");
  }
  if (op.y == nullptr)
  {
    checkCusparse(cusparseCreateDnVec(
                      &op.y, y.size(), y.data(), CUDA_R_64F),
                  "cusparseCreateDnVec(output) failed");
  }
  else
  {
    checkCusparse(cusparseDnVecSetValues(op.y, y.data()),
                  "cusparseDnVecSetValues(output) failed");
  }
}

void spmv(const DeviceCsrMatrix& mat,
          DeviceConstVectorView  x,
          DeviceVectorView       y,
          CudaContext&           ctx,
          Real                   alpha,
          Real                   beta,
          bool                   transpose)
{
  CsrSpmvState&               state = spmvState(ctx);
  std::lock_guard<std::mutex> lock(state.mutex);

  auto& mat_state = state.matrices[matrixKey(mat)];
  if (!mat_state)
  {
    mat_state = std::make_unique<MatrixSpmvState>();
  }
  SpmvOperation& operation = transpose
                                 ? mat_state->apply_transpose
                                 : mat_state->apply;
  ensureDescriptors(operation, mat, x, y);

  const auto  op             = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                         : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto        handle         = cusparseHandle(ctx.stream());
  std::size_t workspace_size = 0;
  checkCusparse(cusparseSpMV_bufferSize(handle,
                                        op,
                                        &alpha,
                                        operation.matrix,
                                        operation.x,
                                        &beta,
                                        operation.y,
                                        CUDA_R_64F,
                                        CUSPARSE_SPMV_CSR_ALG1,
                                        &workspace_size),
                "cusparseSpMV_bufferSize failed");
  if (workspace_size > operation.workspace_capacity)
  {
    device::release(operation.workspace);
    operation.workspace          = device::allocate(workspace_size);
    operation.workspace_capacity = workspace_size;
    operation.preprocessed       = false;
  }
  if (!operation.preprocessed)
  {
    checkCusparse(cusparseSpMV_preprocess(handle,
                                          op,
                                          &alpha,
                                          operation.matrix,
                                          operation.x,
                                          &beta,
                                          operation.y,
                                          CUDA_R_64F,
                                          CUSPARSE_SPMV_CSR_ALG1,
                                          operation.workspace),
                  "cusparseSpMV_preprocess failed");
    operation.preprocessed = true;
  }
  checkCusparse(cusparseSpMV(handle,
                             op,
                             &alpha,
                             operation.matrix,
                             operation.x,
                             &beta,
                             operation.y,
                             CUDA_R_64F,
                             CUSPARSE_SPMV_CSR_ALG1,
                             operation.workspace),
                "cusparseSpMV failed");
}
} // namespace

void apply(const DeviceCsrMatrix& mat,
           DeviceConstVectorView  x,
           DeviceVectorView       y,
           CudaContext&           ctx,
           Real                   alpha,
           Real                   beta)
{
  checkApply(mat, x, y, false);
  if (mat.rows() == 0 || mat.nnz() == 0 || alpha == 0.0)
  {
    scaleOutput(y, beta, ctx);
    return;
  }
  spmv(mat, x, y, ctx, alpha, beta, false);
}

void applyT(const DeviceCsrMatrix& mat,
            DeviceConstVectorView  x,
            DeviceVectorView       y,
            CudaContext&           ctx,
            Real                   alpha,
            Real                   beta)
{
  checkApply(mat, x, y, true);
  if (mat.cols() == 0 || mat.nnz() == 0 || alpha == 0.0)
  {
    scaleOutput(y, beta, ctx);
    return;
  }
  spmv(mat, x, y, ctx, alpha, beta, true);
}

} // namespace femx
