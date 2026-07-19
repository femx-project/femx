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
                  CudaContext& ctx)
{
  require(workspace != nullptr,
          "ReSolve Device transpose workspace is missing");
  require(rows >= 0 && cols >= 0 && nnz >= 0
              && (rows == 0 || src_row_ptr != nullptr)
              && (cols == 0 || dst_row_ptr != nullptr)
              && (nnz == 0
                  || (src_vals != nullptr && src_col_ind != nullptr
                      && dst_vals != nullptr && dst_col_ind != nullptr)),
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
                    CUSPARSE_ACTION_NUMERIC,
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
                                   CUSPARSE_ACTION_NUMERIC,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUSPARSE_CSR2CSC_ALG1,
                                   work.buffer),
                "cusparseCsr2cscEx2 failed");
}

} // namespace femx::linalg::detail
