#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <femx/common/Checks.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::linalg::detail
{
namespace
{
void checkCusparse(cusparseStatus_t status, const char* op)
{
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error(std::string(op) + ": "
                             + cusparseGetErrorString(status));
  }
}

struct CsrTransposeWorkspace
{
  cusparseHandle_t handle{nullptr};
  void*            buffer{nullptr};
  std::size_t      capacity{0};

  CsrTransposeWorkspace()
  {
    checkCusparse(cusparseCreate(&handle),
                  "cusparseCreate for ReSolve transpose failed");
  }

  ~CsrTransposeWorkspace()
  {
    device::release(buffer);
    if (handle != nullptr)
    {
      cusparseDestroy(handle);
    }
  }
};

__global__ void buildTrMapKernel(Index        rows,
                                 const Index* src_row_ptr,
                                 const Index* src_col_ind,
                                 const Index* tr_row_ptr,
                                 const Index* tr_col_ind,
                                 Index*       src_to_tr)
{
  const Index stride = static_cast<Index>(blockDim.x * gridDim.x);
  for (Index row = static_cast<Index>(blockIdx.x * blockDim.x
                                      + threadIdx.x);
       row < rows;
       row += stride)
  {
    for (Index k = src_row_ptr[row]; k < src_row_ptr[row + 1]; ++k)
    {
      const Index tr_row = src_col_ind[k];
      Index       rank   = 0;
      for (Index prev = src_row_ptr[row]; prev < k; ++prev)
      {
        rank += src_col_ind[prev] == tr_row ? 1 : 0;
      }
      for (Index tr_k = tr_row_ptr[tr_row];
           tr_k < tr_row_ptr[tr_row + 1];
           ++tr_k)
      {
        if (tr_col_ind[tr_k] == row)
        {
          if (rank == 0)
          {
            src_to_tr[k] = tr_k;
            break;
          }
          --rank;
        }
      }
    }
  }
}

__global__ void updateTrValsKernel(Index        nnz,
                                   const Real*  src_vals,
                                   const Index* src_to_tr,
                                   Real*        tr_vals)
{
  const Index stride = static_cast<Index>(blockDim.x * gridDim.x);
  for (Index k = static_cast<Index>(blockIdx.x * blockDim.x
                                    + threadIdx.x);
       k < nnz;
       k += stride)
  {
    tr_vals[src_to_tr[k]] = src_vals[k];
  }
}

} // namespace

void* createCsrTransposeWorkspace()
{
  return new CsrTransposeWorkspace;
}

void destroyCsrTransposeWorkspace(void* workspace) noexcept
{
  delete static_cast<CsrTransposeWorkspace*>(workspace);
}

void transposeCsr(void*        workspace,
                  Index        rows,
                  Index        cols,
                  Index        nnz,
                  const Real*  src_vals,
                  const Index* src_row_ptr,
                  const Index* src_col_ind,
                  Real*        dst_vals,
                  Index*       dst_row_ptr,
                  Index*       dst_col_ind,
                  Index*       src_to_tr,
                  bool         rebuild_graph,
                  CudaContext& ctx)
{
  require(workspace != nullptr,
          "ReSolve Device transpose workspace is missing");
  require(rows >= 0 && cols >= 0 && nnz >= 0
              && (rows == 0 || src_row_ptr != nullptr)
              && (cols == 0 || dst_row_ptr != nullptr)
              && (nnz == 0
                  || (src_vals != nullptr && src_col_ind != nullptr
                      && dst_vals != nullptr && dst_col_ind != nullptr
                      && src_to_tr != nullptr)),
          "ReSolve Device transpose has incomplete CSR storage");
  if (rows == 0 || cols == 0 || nnz == 0)
  {
    return;
  }

  auto& work = *static_cast<CsrTransposeWorkspace*>(workspace);
  checkCusparse(
      cusparseSetStream(work.handle,
                        static_cast<cudaStream_t>(ctx.stream())),
      "cusparseSetStream for ReSolve transpose failed");

  const auto stream = static_cast<cudaStream_t>(ctx.stream());
  if (rebuild_graph)
  {
    std::size_t size = 0;
    checkCusparse(cusparseCsr2cscEx2_bufferSize(
                      work.handle,
                      rows,
                      cols,
                      nnz,
                      src_vals,
                      src_row_ptr,
                      src_col_ind,
                      dst_vals,
                      dst_row_ptr,
                      dst_col_ind,
                      CUDA_R_64F,
                      CUSPARSE_ACTION_SYMBOLIC,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUSPARSE_CSR2CSC_ALG1,
                      &size),
                  "cusparseCsr2cscEx2_bufferSize failed");
    if (size > work.capacity)
    {
      device::release(work.buffer);
      work.buffer   = device::allocate(size);
      work.capacity = size;
    }
    checkCusparse(cusparseCsr2cscEx2(work.handle,
                                     rows,
                                     cols,
                                     nnz,
                                     src_vals,
                                     src_row_ptr,
                                     src_col_ind,
                                     dst_vals,
                                     dst_row_ptr,
                                     dst_col_ind,
                                     CUDA_R_64F,
                                     CUSPARSE_ACTION_SYMBOLIC,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUSPARSE_CSR2CSC_ALG1,
                                     work.buffer),
                  "cusparseCsr2cscEx2 symbolic transpose failed");

    constexpr unsigned int threads = 128;
    const unsigned int     blocks  = static_cast<unsigned int>(
        (rows + static_cast<Index>(threads) - 1)
        / static_cast<Index>(threads));
    buildTrMapKernel<<<blocks, threads, 0, stream>>>(rows,
                                                     src_row_ptr,
                                                     src_col_ind,
                                                     dst_row_ptr,
                                                     dst_col_ind,
                                                     src_to_tr);
    device::checkLastError();
  }

  constexpr unsigned int threads = 256;
  const unsigned int     blocks  = static_cast<unsigned int>(
      (nnz + static_cast<Index>(threads) - 1)
      / static_cast<Index>(threads));
  updateTrValsKernel<<<blocks, threads, 0, stream>>>(
      nnz, src_vals, src_to_tr, dst_vals);
  device::checkLastError();
}

} // namespace femx::linalg::detail
