#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include <cublas_v2.h>
#include <femx/common/Cuda.hpp>
#include <femx/common/Device.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaHandles.hpp>
#include <femx/linalg/cuda/CudaMatrixHandler.hpp>

namespace femx::linalg
{
namespace
{
constexpr int kThreads = 256;

using detail::checkCublas;
using detail::checkCusparse;

struct SpmvOperation
{
  cusparseSpMatDescr_t mat{nullptr};
  cusparseDnVecDescr_t x{nullptr};
  cusparseDnVecDescr_t y{nullptr};
  void*                workspace{nullptr};
  std::size_t          workspace_capacity{0};
  bool                 preprocessed{false};

  ~SpmvOperation()
  {
    device::release(workspace);
    if (y != nullptr)
    {
      cusparseDestroyDnVec(y);
    }
    if (x != nullptr)
    {
      cusparseDestroyDnVec(x);
    }
    if (mat != nullptr)
    {
      cusparseDestroySpMat(mat);
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
  std::uint64_t       src_layout{0};
  DeviceCsrPattern    pattern;
  DeviceVector<Index> src_to_trans;
};

struct CsrState
{
  ~CsrState()
  {
    device::release(trans_workspace);
  }

  std::mutex                                      mutex;
  std::vector<std::unique_ptr<MatrixEntry>>       mats;
  std::vector<std::unique_ptr<CsrTransposeEntry>> trans_entries;
  void*                                           trans_workspace{nullptr};
  std::size_t                                     trans_workspace_capacity{0};
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
                                        const Index* src_row_ptr,
                                        const Index* src_col_ind,
                                        const Index* trans_row_ptr,
                                        const Index* trans_col_ind,
                                        Index*       src_to_trans)
{
  const Index stride = static_cast<Index>(blockDim.x * gridDim.x);
  for (Index row = static_cast<Index>(blockIdx.x * blockDim.x
                                      + threadIdx.x);
       row < rows;
       row += stride)
  {
    for (Index k = src_row_ptr[row]; k < src_row_ptr[row + 1]; ++k)
    {
      const Index trans_row = src_col_ind[k];
      Index       rank      = 0;
      for (Index previous = src_row_ptr[row]; previous < k; ++previous)
      {
        rank += src_col_ind[previous] == trans_row ? 1 : 0;
      }
      for (Index trans_idx = trans_row_ptr[trans_row];
           trans_idx < trans_row_ptr[trans_row + 1];
           ++trans_idx)
      {
        if (trans_col_ind[trans_idx] == row)
        {
          if (rank == 0)
          {
            src_to_trans[k] = trans_idx;
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
    const Real*  src_vals,
    const Index* src_to_trans,
    Real*        trans_vals)
{
  const Index stride = static_cast<Index>(blockDim.x * gridDim.x);
  for (Index k = static_cast<Index>(blockIdx.x * blockDim.x
                                    + threadIdx.x);
       k < nnz;
       k += stride)
  {
    trans_vals[src_to_trans[k]] = src_vals[k];
  }
}

void checkCsrMatvec(const DeviceCsrMatrix&       mat,
                    DeviceVectorView<const Real> dir,
                    DeviceVectorView<Real>       out,
                    bool                         trans)
{
  require(dir.isValid(), "CSR matvec has an invalid input view");
  require(out.isValid(), "CSR matvec has an invalid output view");
  const Index in_size  = trans ? mat.rows() : mat.cols();
  const Index out_size = trans ? mat.cols() : mat.rows();
  require(dir.size() == in_size && out.size() == out_size,
          "CSR matvec vector size mismatch");
  require(mat.rows() == 0 || mat.rowPtrData() != nullptr,
          "CSR matvec has no Device row offsets");
  require(mat.nnz() == 0
              || (mat.colIndData() != nullptr && mat.valsData() != nullptr),
          "CSR matvec has incomplete Device storage");
  require(!femx::detail::overlaps(dir, out),
          "CSR matvec does not support in-place vectors");
  require(!femx::detail::overlaps(mat.valsData(),
                                  mat.nnz(),
                                  out.data(),
                                  out.size()),
          "CSR matvec output aliases matrix values");
}

void checkDenseMatvec(DeviceMatrixView<const Real> mat,
                      DeviceVectorView<const Real> dir,
                      DeviceVectorView<Real>       out,
                      bool                         trans)
{
  const Index in_size  = trans ? mat.rows() : mat.cols();
  const Index out_size = trans ? mat.cols() : mat.rows();
  require(mat.rows() >= 0 && mat.cols() >= 0 && dir.size() == in_size
              && out.size() == out_size
              && (mat.rows() * mat.cols() == 0 || mat.data() != nullptr),
          "Device dense matvec received incompatible storage");
  require(!femx::detail::overlaps(dir, out)
              && !femx::detail::overlaps(mat.data(),
                                         mat.rows() * mat.cols(),
                                         out.data(),
                                         out.size()),
          "Device dense matvec does not support aliased vectors");
}

void scaleOutput(DeviceVectorView<Real> out, Real beta, CudaContext& ctx)
{
  if (out.empty() || beta == 1.0)
  {
    return;
  }
  if (beta == 0.0)
  {
    device::zero(out.data(),
                 static_cast<std::size_t>(out.size()) * sizeof(Real),
                 ctx.stream());
    return;
  }
  scaleKernel<<<cuda::numBlocks(out.size(), kThreads),
                kThreads,
                0,
                static_cast<cudaStream_t>(ctx.stream())>>>(
      out.size(), beta, out.data());
  cuda::checkLastError();
}

CsrState& csrState(CudaContext& ctx)
{
  auto& storage = detail::cudaSparseState(ctx);
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
      state.mats.begin(),
      state.mats.end(),
      [&mat](const std::unique_ptr<MatrixEntry>& entry)
      { return entry->matches(mat); });
  if (iter != state.mats.end())
  {
    return **iter;
  }

  state.mats.push_back(std::make_unique<MatrixEntry>(mat));
  return *state.mats.back();
}

CsrTransposeEntry* findTransposeEntry(CsrState&              state,
                                      const DeviceCsrMatrix& src)
{
  const auto iter = std::find_if(
      state.trans_entries.begin(),
      state.trans_entries.end(),
      [&src](const std::unique_ptr<CsrTransposeEntry>& entry)
      { return entry->src_layout == src.pattern().layoutId(); });
  return iter == state.trans_entries.end() ? nullptr : iter->get();
}

void ensureDescriptors(SpmvOperation&               op,
                       const DeviceCsrMatrix&       mat,
                       DeviceVectorView<const Real> dir,
                       DeviceVectorView<Real>       out)
{
  if (op.mat == nullptr)
  {
    checkCusparse(
        cusparseCreateCsr(&op.mat,
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
                                      dir.size(),
                                      const_cast<Real*>(dir.data()),
                                      CUDA_R_64F),
                  "cusparseCreateDnVec(input) failed");
  }
  else
  {
    checkCusparse(cusparseDnVecSetValues(op.x,
                                         const_cast<Real*>(dir.data())),
                  "cusparseDnVecSetValues(input) failed");
  }
  if (op.y == nullptr)
  {
    checkCusparse(cusparseCreateDnVec(&op.y,
                                      out.size(),
                                      out.data(),
                                      CUDA_R_64F),
                  "cusparseCreateDnVec(output) failed");
  }
  else
  {
    checkCusparse(cusparseDnVecSetValues(op.y, out.data()),
                  "cusparseDnVecSetValues(output) failed");
  }
}

void spmv(const DeviceCsrMatrix&       mat,
          DeviceVectorView<const Real> dir,
          DeviceVectorView<Real>       out,
          CudaContext&                 ctx,
          Real                         alpha,
          Real                         beta,
          bool                         trans)
{
  CsrState&                   state = csrState(ctx);
  std::lock_guard<std::mutex> lock(state.mutex);
  MatrixEntry&                entry   = findOrCreateEntry(state, mat);
  SpmvOperation&              spmv_op = trans ? entry.matvecT : entry.matvec;
  ensureDescriptors(spmv_op, mat, dir, out);

  const auto  cusparse_op    = trans ? CUSPARSE_OPERATION_TRANSPOSE
                                     : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto        handle         = detail::cusparseHandle(ctx);
  std::size_t workspace_size = 0;
  checkCusparse(cusparseSpMV_bufferSize(handle,
                                        cusparse_op,
                                        &alpha,
                                        spmv_op.mat,
                                        spmv_op.x,
                                        &beta,
                                        spmv_op.y,
                                        CUDA_R_64F,
                                        CUSPARSE_SPMV_CSR_ALG1,
                                        &workspace_size),
                "cusparseSpMV_bufferSize failed");
  if (workspace_size > spmv_op.workspace_capacity)
  {
    device::release(spmv_op.workspace);
    spmv_op.workspace          = device::allocate(workspace_size);
    spmv_op.workspace_capacity = workspace_size;
    spmv_op.preprocessed       = false;
  }
  if (!spmv_op.preprocessed)
  {
    checkCusparse(cusparseSpMV_preprocess(handle,
                                          cusparse_op,
                                          &alpha,
                                          spmv_op.mat,
                                          spmv_op.x,
                                          &beta,
                                          spmv_op.y,
                                          CUDA_R_64F,
                                          CUSPARSE_SPMV_CSR_ALG1,
                                          spmv_op.workspace),
                  "cusparseSpMV_preprocess failed");
    spmv_op.preprocessed = true;
  }
  checkCusparse(cusparseSpMV(handle,
                             cusparse_op,
                             &alpha,
                             spmv_op.mat,
                             spmv_op.x,
                             &beta,
                             spmv_op.y,
                             CUDA_R_64F,
                             CUSPARSE_SPMV_CSR_ALG1,
                             spmv_op.workspace),
                "cusparseSpMV failed");
}

} // namespace

void CudaMatrixHandler::transpose(const DeviceCsrMatrix& src,
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
    DeviceVector<Index> trans_row_ptr(src.cols() + 1);
    DeviceVector<Index> trans_col_ind(src.nnz());
    DeviceVector<Index> src_to_trans(src.nnz());

    auto created        = std::make_unique<CsrTransposeEntry>();
    created->src_layout = src.pattern().layoutId();
    created->pattern    = DeviceCsrPattern(
        src.cols(),
        src.rows(),
        std::move(trans_row_ptr),
        std::move(trans_col_ind),
        femx::detail::newCsrLayoutId());

    created->src_to_trans = std::move(src_to_trans);
    state.trans_entries.push_back(std::move(created));
    entry = state.trans_entries.back().get();
  }

  if (dst.pattern().layoutId() != entry->pattern.layoutId())
  {
    dst = DeviceCsrMatrix(entry->pattern);
  }
  if (src.nnz() == 0)
  {
    return;
  }

  auto handle = detail::cusparseHandle(ctx_);
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

    if (workspace_size > state.trans_workspace_capacity)
    {
      void* replacement = device::allocate(workspace_size);
      device::release(state.trans_workspace);
      state.trans_workspace          = replacement;
      state.trans_workspace_capacity = workspace_size;
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
                      state.trans_workspace),
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
        entry->src_to_trans.data());
    cuda::checkLastError();
  }

  constexpr unsigned int threads = 256;
  updateTransposeValuesKernel<<<cuda::numBlocks(src.nnz(), threads),
                                threads,
                                0,
                                static_cast<cudaStream_t>(ctx_.stream())>>>(
      src.nnz(),
      src.valsData(),
      entry->src_to_trans.data(),
      dst.valsData());
  cuda::checkLastError();
}

void CudaMatrixHandler::matvec(const DeviceCsrMatrix&       mat,
                               DeviceVectorView<const Real> dir,
                               DeviceVectorView<Real>       out,
                               Real                         alpha,
                               Real                         beta) const
{
  checkCsrMatvec(mat, dir, out, false);
  if (mat.rows() == 0 || mat.nnz() == 0 || alpha == 0.0)
  {
    scaleOutput(out, beta, ctx_);
    return;
  }
  spmv(mat, dir, out, ctx_, alpha, beta, false);
}

void CudaMatrixHandler::matvecT(const DeviceCsrMatrix&       mat,
                                DeviceVectorView<const Real> dir,
                                DeviceVectorView<Real>       out,
                                Real                         alpha,
                                Real                         beta) const
{
  checkCsrMatvec(mat, dir, out, true);
  if (mat.cols() == 0 || mat.nnz() == 0 || alpha == 0.0)
  {
    scaleOutput(out, beta, ctx_);
    return;
  }
  spmv(mat, dir, out, ctx_, alpha, beta, true);
}

void CudaMatrixHandler::matvec(DeviceMatrixView<const Real> mat,
                               DeviceVectorView<const Real> dir,
                               DeviceVectorView<Real>       out,
                               Real                         alpha,
                               Real                         beta) const
{
  checkDenseMatvec(mat, dir, out, false);
  if (mat.rows() == 0)
  {
    return;
  }
  if (mat.cols() == 0)
  {
    scaleOutput(out, beta, ctx_);
    return;
  }
  auto handle = detail::cublasHandle(ctx_);
  checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
              "cublasSetPointerMode failed");
  checkCublas(cublasDgemv(handle,
                          CUBLAS_OP_T,
                          mat.cols(),
                          mat.rows(),
                          &alpha,
                          mat.data(),
                          mat.cols(),
                          dir.data(),
                          1,
                          &beta,
                          out.data(),
                          1),
              "cublasDgemv failed");
}

void CudaMatrixHandler::matvecT(DeviceMatrixView<const Real> mat,
                                DeviceVectorView<const Real> dir,
                                DeviceVectorView<Real>       out,
                                Real                         alpha,
                                Real                         beta) const
{
  checkDenseMatvec(mat, dir, out, true);
  if (mat.cols() == 0)
  {
    return;
  }
  if (mat.rows() == 0)
  {
    scaleOutput(out, beta, ctx_);
    return;
  }
  auto handle = detail::cublasHandle(ctx_);
  checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
              "cublasSetPointerMode failed");
  checkCublas(cublasDgemv(handle,
                          CUBLAS_OP_N,
                          mat.cols(),
                          mat.rows(),
                          &alpha,
                          mat.data(),
                          mat.cols(),
                          dir.data(),
                          1,
                          &beta,
                          out.data(),
                          1),
              "cublasDgemv(transpose) failed");
}

} // namespace femx::linalg
