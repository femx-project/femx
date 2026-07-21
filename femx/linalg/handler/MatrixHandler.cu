#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include <cublas_v2.h>
#include <femx/linalg/handler/CudaHandles.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>

namespace femx::detail
{
struct CudaContextAccess
{
  static std::shared_ptr<void>& sparseState(CudaContext& ctx)
  {
    return ctx.sparse_state_;
  }
};
} // namespace femx::detail

namespace femx::linalg
{
namespace
{
constexpr int kThreads = 256;

using detail::checkCublas;
using detail::checkCusparse;

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
    cuda::release(workspace);
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

struct MatrixEntry
{
  const Index* row_ptr{nullptr};
  const Index* col_ind{nullptr};
  const Real*  vals{nullptr};
  Index        rows{0};
  Index        cols{0};
  Index        nnz{0};

  SpmvOperation matvec;
  SpmvOperation matvecT;

  explicit MatrixEntry(const DeviceCsrMatrix& mat) noexcept
    : row_ptr(mat.rowPtrData()),
      col_ind(mat.colIndData()),
      vals(mat.valsData()),
      rows(mat.rows()),
      cols(mat.cols()),
      nnz(mat.nnz())
  {
  }

  bool matches(const DeviceCsrMatrix& mat) const noexcept
  {
    return row_ptr == mat.rowPtrData() && col_ind == mat.colIndData()
           && vals == mat.valsData() && rows == mat.rows()
           && cols == mat.cols() && nnz == mat.nnz();
  }
};

struct CsrTransposeEntry
{
  std::uint64_t     source_layout{0};
  DeviceCsrPattern  pattern;
  DeviceIndexVector source_to_transpose;
};

struct CsrState
{
  ~CsrState()
  {
    cuda::release(transpose_workspace);
  }

  std::mutex                                      mutex;
  std::vector<std::unique_ptr<MatrixEntry>>       matrices;
  std::vector<std::unique_ptr<CsrTransposeEntry>> transposes;
  void*                                           transpose_workspace{nullptr};
  std::size_t                                     transpose_workspace_capacity{0};
};

__global__ void scaleKernel(Index size, Real scale, Real* vals)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    vals[i] *= scale;
  }
}

__global__ void buildTransposeMapKernel(Index        rows,
                                        const Index* source_row_ptr,
                                        const Index* source_col_ind,
                                        const Index* transpose_row_ptr,
                                        const Index* transpose_col_ind,
                                        Index*       source_to_transpose)
{
  const Index stride = static_cast<Index>(blockDim.x * gridDim.x);
  for (Index row = static_cast<Index>(blockIdx.x * blockDim.x
                                      + threadIdx.x);
       row < rows;
       row += stride)
  {
    for (Index k = source_row_ptr[row]; k < source_row_ptr[row + 1]; ++k)
    {
      const Index transpose_row = source_col_ind[k];
      Index       rank          = 0;
      for (Index previous = source_row_ptr[row]; previous < k; ++previous)
      {
        rank += source_col_ind[previous] == transpose_row ? 1 : 0;
      }
      for (Index transpose_index = transpose_row_ptr[transpose_row];
           transpose_index < transpose_row_ptr[transpose_row + 1];
           ++transpose_index)
      {
        if (transpose_col_ind[transpose_index] == row)
        {
          if (rank == 0)
          {
            source_to_transpose[k] = transpose_index;
            break;
          }
          --rank;
        }
      }
    }
  }
}

__global__ void updateTransposeValuesKernel(
    Index        nnz,
    const Real*  source_vals,
    const Index* source_to_transpose,
    Real*        transpose_vals)
{
  const Index stride = static_cast<Index>(blockDim.x * gridDim.x);
  for (Index k = static_cast<Index>(blockIdx.x * blockDim.x
                                    + threadIdx.x);
       k < nnz;
       k += stride)
  {
    transpose_vals[source_to_transpose[k]] = source_vals[k];
  }
}

void checkCsrMatvec(const DeviceCsrMatrix& mat,
                    DeviceConstVectorView  x,
                    DeviceVectorView       y,
                    bool                   transpose)
{
  require(x.isValid(), "CSR matvec has an invalid input view");
  require(y.isValid(), "CSR matvec has an invalid output view");
  const Index in_size  = transpose ? mat.rows() : mat.cols();
  const Index out_size = transpose ? mat.cols() : mat.rows();
  require(x.size() == in_size && y.size() == out_size,
          "CSR matvec vector size mismatch");
  require(mat.rows() == 0 || mat.rowPtrData() != nullptr,
          "CSR matvec has no Device row offsets");
  require(mat.nnz() == 0
              || (mat.colIndData() != nullptr && mat.valsData() != nullptr),
          "CSR matvec has incomplete Device storage");
  require(!femx::detail::overlaps(x, y),
          "CSR matvec does not support in-place vectors");
  require(!femx::detail::overlaps(mat.valsData(), mat.nnz(), y.data(), y.size()),
          "CSR matvec output aliases matrix values");
}

void checkDenseMatvec(DeviceMatrixView<const Real> mat,
                      DeviceConstVectorView        x,
                      DeviceVectorView             y,
                      bool                         transpose)
{
  const Index in_size  = transpose ? mat.rows() : mat.cols();
  const Index out_size = transpose ? mat.cols() : mat.rows();
  require(mat.rows() >= 0 && mat.cols() >= 0 && x.size() == in_size
              && y.size() == out_size
              && (mat.rows() * mat.cols() == 0 || mat.data() != nullptr),
          "Device dense matvec received incompatible storage");
  require(!femx::detail::overlaps(x, y)
              && !femx::detail::overlaps(mat.data(),
                                         mat.rows() * mat.cols(),
                                         y.data(),
                                         y.size()),
          "Device dense matvec does not support aliased vectors");
}

void scaleOutput(DeviceVectorView y, Real beta, CudaContext& ctx)
{
  if (y.empty() || beta == 1.0)
  {
    return;
  }
  if (beta == 0.0)
  {
    cuda::zero(y.data(),
               static_cast<std::size_t>(y.size()) * sizeof(Real),
               ctx.stream());
    return;
  }
  scaleKernel<<<cuda::numBlocks(y.size(), kThreads),
                kThreads,
                0,
                static_cast<cudaStream_t>(ctx.stream())>>>(
      y.size(), beta, y.data());
  cuda::checkLastError();
}

CsrState& csrState(CudaContext& ctx)
{
  auto& storage = femx::detail::CudaContextAccess::sparseState(ctx);
  if (!storage)
  {
    storage = std::shared_ptr<void>(
        new CsrState,
        [](void* state)
        { delete static_cast<CsrState*>(state); });
  }
  return *static_cast<CsrState*>(storage.get());
}

MatrixEntry& findOrCreateEntry(CsrState&              state,
                               const DeviceCsrMatrix& mat)
{
  const auto iter = std::find_if(
      state.matrices.begin(),
      state.matrices.end(),
      [&mat](const std::unique_ptr<MatrixEntry>& entry)
      { return entry->matches(mat); });
  if (iter != state.matrices.end())
  {
    return **iter;
  }

  state.matrices.push_back(std::make_unique<MatrixEntry>(mat));
  return *state.matrices.back();
}

CsrTransposeEntry* findTransposeEntry(CsrState&              state,
                                      const DeviceCsrMatrix& src)
{
  const auto iter = std::find_if(
      state.transposes.begin(),
      state.transposes.end(),
      [&src](const std::unique_ptr<CsrTransposeEntry>& entry)
      { return entry->source_layout == src.pattern().layoutId(); });
  return iter == state.transposes.end() ? nullptr : iter->get();
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
    checkCusparse(cusparseCreateDnVec(&op.x,
                                      x.size(),
                                      const_cast<Real*>(x.data()),
                                      CUDA_R_64F),
                  "cusparseCreateDnVec(input) failed");
  }
  else
  {
    checkCusparse(cusparseDnVecSetValues(op.x,
                                         const_cast<Real*>(x.data())),
                  "cusparseDnVecSetValues(input) failed");
  }
  if (op.y == nullptr)
  {
    checkCusparse(cusparseCreateDnVec(&op.y,
                                      y.size(),
                                      y.data(),
                                      CUDA_R_64F),
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
  CsrState&                   state = csrState(ctx);
  std::lock_guard<std::mutex> lock(state.mutex);
  MatrixEntry&                entry     = findOrCreateEntry(state, mat);
  SpmvOperation&              operation = transpose ? entry.matvecT
                                                    : entry.matvec;
  ensureDescriptors(operation, mat, x, y);

  const auto  op             = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                         : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto        handle         = detail::cusparseHandle(ctx.stream());
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
    cuda::release(operation.workspace);
    operation.workspace          = cuda::allocate(workspace_size);
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

void MatrixHandler<CudaCsrBackend>::transpose(
    const DeviceCsrMatrix& src,
    DeviceCsrMatrix&       dst) const
{
  require(&src != &dst, "CSR transpose does not support in-place output");
  require(src.cols() != std::numeric_limits<Index>::max(),
          "CSR transpose row count is too large");

  CsrState&                   state = csrState(ctx_);
  std::lock_guard<std::mutex> lock(state.mutex);
  CsrTransposeEntry*          entry           = findTransposeEntry(state, src);
  const bool                  rebuild_pattern = entry == nullptr;

  if (rebuild_pattern)
  {
    DeviceIndexVector transpose_row_ptr(src.cols() + 1);
    DeviceIndexVector transpose_col_ind(src.nnz());
    DeviceIndexVector source_to_transpose(src.nnz());

    auto created           = std::make_unique<CsrTransposeEntry>();
    created->source_layout = src.pattern().layoutId();
    created->pattern       = DeviceCsrPattern(
        src.cols(),
        src.rows(),
        std::move(transpose_row_ptr),
        std::move(transpose_col_ind),
        femx::detail::newCsrLayoutId());
    created->source_to_transpose = std::move(source_to_transpose);
    state.transposes.push_back(std::move(created));
    entry = state.transposes.back().get();
  }

  if (dst.pattern().layoutId() != entry->pattern.layoutId())
  {
    dst = DeviceCsrMatrix(entry->pattern);
  }
  if (src.nnz() == 0)
  {
    return;
  }

  auto handle = detail::cusparseHandle(ctx_.stream());
  if (rebuild_pattern)
  {
    std::size_t workspace_size = 0;
    checkCusparse(cusparseCsr2cscEx2_bufferSize(
                      handle,
                      src.rows(),
                      src.cols(),
                      src.nnz(),
                      src.valsData(),
                      src.rowPtrData(),
                      src.colIndData(),
                      dst.valsData(),
                      const_cast<Index*>(dst.rowPtrData()),
                      const_cast<Index*>(dst.colIndData()),
                      CUDA_R_64F,
                      CUSPARSE_ACTION_SYMBOLIC,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUSPARSE_CSR2CSC_ALG1,
                      &workspace_size),
                  "cusparseCsr2cscEx2_bufferSize failed");
    if (workspace_size > state.transpose_workspace_capacity)
    {
      void* replacement = cuda::allocate(workspace_size);
      cuda::release(state.transpose_workspace);
      state.transpose_workspace          = replacement;
      state.transpose_workspace_capacity = workspace_size;
    }
    checkCusparse(cusparseCsr2cscEx2(
                      handle,
                      src.rows(),
                      src.cols(),
                      src.nnz(),
                      src.valsData(),
                      src.rowPtrData(),
                      src.colIndData(),
                      dst.valsData(),
                      const_cast<Index*>(dst.rowPtrData()),
                      const_cast<Index*>(dst.colIndData()),
                      CUDA_R_64F,
                      CUSPARSE_ACTION_SYMBOLIC,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUSPARSE_CSR2CSC_ALG1,
                      state.transpose_workspace),
                  "cusparseCsr2cscEx2 symbolic transpose failed");

    constexpr unsigned int threads = 128;
    buildTransposeMapKernel<<<cuda::numBlocks(src.rows(), threads),
                              threads,
                              0,
                              static_cast<cudaStream_t>(ctx_.stream())>>>(
        src.rows(),
        src.rowPtrData(),
        src.colIndData(),
        dst.rowPtrData(),
        dst.colIndData(),
        entry->source_to_transpose.data());
    cuda::checkLastError();
  }

  constexpr unsigned int threads = 256;
  updateTransposeValuesKernel<<<cuda::numBlocks(src.nnz(), threads),
                                threads,
                                0,
                                static_cast<cudaStream_t>(ctx_.stream())>>>(
      src.nnz(),
      src.valsData(),
      entry->source_to_transpose.data(),
      dst.valsData());
  cuda::checkLastError();
}

void MatrixHandler<CudaCsrBackend>::matvec(const DeviceCsrMatrix& mat,
                                           DeviceConstVectorView  x,
                                           DeviceVectorView       y,
                                           Real                   alpha,
                                           Real                   beta) const
{
  checkCsrMatvec(mat, x, y, false);
  if (mat.rows() == 0 || mat.nnz() == 0 || alpha == 0.0)
  {
    scaleOutput(y, beta, ctx_);
    return;
  }
  spmv(mat, x, y, ctx_, alpha, beta, false);
}

void MatrixHandler<CudaCsrBackend>::matvecT(const DeviceCsrMatrix& mat,
                                            DeviceConstVectorView  x,
                                            DeviceVectorView       y,
                                            Real                   alpha,
                                            Real                   beta) const
{
  checkCsrMatvec(mat, x, y, true);
  if (mat.cols() == 0 || mat.nnz() == 0 || alpha == 0.0)
  {
    scaleOutput(y, beta, ctx_);
    return;
  }
  spmv(mat, x, y, ctx_, alpha, beta, true);
}

void MatrixHandler<CudaCsrBackend>::matvec(DeviceMatrixView<const Real> mat,
                                           DeviceConstVectorView        x,
                                           DeviceVectorView             y,
                                           Real                         alpha,
                                           Real                         beta) const
{
  checkDenseMatvec(mat, x, y, false);
  if (mat.rows() == 0)
  {
    return;
  }
  if (mat.cols() == 0)
  {
    scaleOutput(y, beta, ctx_);
    return;
  }
  auto handle = detail::cublasHandle(ctx_.stream());
  checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
              "cublasSetPointerMode failed");
  checkCublas(cublasDgemv(handle,
                          CUBLAS_OP_T,
                          mat.cols(),
                          mat.rows(),
                          &alpha,
                          mat.data(),
                          mat.cols(),
                          x.data(),
                          1,
                          &beta,
                          y.data(),
                          1),
              "cublasDgemv failed");
}

void MatrixHandler<CudaCsrBackend>::matvecT(DeviceMatrixView<const Real> mat,
                                            DeviceConstVectorView        x,
                                            DeviceVectorView             y,
                                            Real                         alpha,
                                            Real                         beta) const
{
  checkDenseMatvec(mat, x, y, true);
  if (mat.cols() == 0)
  {
    return;
  }
  if (mat.rows() == 0)
  {
    scaleOutput(y, beta, ctx_);
    return;
  }
  auto handle = detail::cublasHandle(ctx_.stream());
  checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
              "cublasSetPointerMode failed");
  checkCublas(cublasDgemv(handle,
                          CUBLAS_OP_N,
                          mat.cols(),
                          mat.rows(),
                          &alpha,
                          mat.data(),
                          mat.cols(),
                          x.data(),
                          1,
                          &beta,
                          y.data(),
                          1),
              "cublasDgemv(transpose) failed");
}

} // namespace femx::linalg
