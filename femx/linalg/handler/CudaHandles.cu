#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include <femx/linalg/handler/CudaHandles.hpp>

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

namespace
{

class CublasHandle
{
public:
  CublasHandle()
  {
    checkCublas(cublasCreate(&handle_), "cublasCreate failed");
  }

  ~CublasHandle()
  {
    if (handle_ != nullptr)
    {
      cublasDestroy(handle_);
    }
  }

  cublasHandle_t get(void* stream)
  {
    checkCublas(cublasSetStream(handle_, static_cast<cudaStream_t>(stream)),
                "cublasSetStream failed");
    return handle_;
  }

private:
  cublasHandle_t handle_{nullptr};
};

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

} // namespace

cublasHandle_t cublasHandle(void* stream)
{
  thread_local CublasHandle handle;
  return handle.get(stream);
}

cusparseHandle_t cusparseHandle(void* stream)
{
  thread_local CusparseHandle handle;
  return handle.get(stream);
}

} // namespace femx::linalg::detail
