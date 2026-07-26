#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaHandles.hpp>

namespace femx::linalg::detail
{

void checkCublas(cublasStatus_t status, const char* operation)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    throw std::runtime_error(std::string(operation) + ": "
                             + cublasGetStatusString(status));
  }
}

void checkCusparse(cusparseStatus_t status, const char* operation)
{
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error(std::string(operation) + ": "
                             + cusparseGetErrorString(status));
  }
}

struct CudaContextAccess
{
  static CudaHandles& handles(CudaContext& ctx)
  {
    return *ctx.handles_;
  }

  static std::shared_ptr<void>& sparseState(CudaContext& ctx)
  {
    return ctx.sparse_state_;
  }
};

CudaHandles::CudaHandles(void* stream)
{
  cublasHandle_t cublas = nullptr;
  checkCublas(cublasCreate(&cublas), "cublasCreate failed");
  try
  {
    checkCublas(cublasSetStream(cublas, static_cast<cudaStream_t>(stream)),
                "cublasSetStream failed");

    cusparseHandle_t cusparse = nullptr;
    checkCusparse(cusparseCreate(&cusparse), "cusparseCreate failed");
    try
    {
      checkCusparse(
          cusparseSetStream(cusparse, static_cast<cudaStream_t>(stream)),
          "cusparseSetStream failed");
    }
    catch (...)
    {
      cusparseDestroy(cusparse);
      throw;
    }

    cublas_   = cublas;
    cusparse_ = cusparse;
  }
  catch (...)
  {
    cublasDestroy(cublas);
    throw;
  }
}

CudaHandles::~CudaHandles()
{
  if (cusparse_ != nullptr)
  {
    cusparseDestroy(static_cast<cusparseHandle_t>(cusparse_));
  }
  if (cublas_ != nullptr)
  {
    cublasDestroy(static_cast<cublasHandle_t>(cublas_));
  }
}

cublasHandle_t CudaHandles::cublas() const noexcept
{
  return static_cast<cublasHandle_t>(cublas_);
}

cusparseHandle_t CudaHandles::cusparse() const noexcept
{
  return static_cast<cusparseHandle_t>(cusparse_);
}

std::unique_ptr<CudaHandles> makeCudaHandles(void* stream)
{
  return std::make_unique<CudaHandles>(stream);
}

std::shared_ptr<void>& cudaSparseState(CudaContext& ctx)
{
  return CudaContextAccess::sparseState(ctx);
}

cublasHandle_t cublasHandle(CudaContext& ctx)
{
  return CudaContextAccess::handles(ctx).cublas();
}

cusparseHandle_t cusparseHandle(CudaContext& ctx)
{
  return CudaContextAccess::handles(ctx).cusparse();
}

} // namespace femx::linalg::detail
