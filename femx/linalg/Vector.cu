#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <femx/common/Checks.hpp>
#include <femx/linalg/Vector.hpp>

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

void checkCublas(cublasStatus_t status, const char* op)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    throw std::runtime_error(std::string(op) + ": "
                             + cublasGetStatusString(status));
  }
}

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
    checkCublas(cublasSetStream(
                    handle_, static_cast<cudaStream_t>(stream)),
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

class SparseVectorDescriptor
{
public:
  SparseVectorDescriptor(Index        size,
                         Index        nnz,
                         const Index* indices,
                         Real*        vals)
  {
    checkCusparse(cusparseCreateSpVec(&descriptor_,
                                      size,
                                      nnz,
                                      const_cast<Index*>(indices),
                                      vals,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_64F),
                  "cusparseCreateSpVec failed");
  }

  SparseVectorDescriptor(Index        size,
                         Index        nnz,
                         const Index* indices,
                         const Real*  vals)
  {
    checkCusparse(cusparseCreateConstSpVec(&const_descriptor_,
                                           size,
                                           nnz,
                                           indices,
                                           vals,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUDA_R_64F),
                  "cusparseCreateConstSpVec failed");
  }

  ~SparseVectorDescriptor()
  {
    if (descriptor_ != nullptr)
    {
      cusparseDestroySpVec(descriptor_);
    }
    if (const_descriptor_ != nullptr)
    {
      cusparseDestroySpVec(const_descriptor_);
    }
  }

  cusparseSpVecDescr_t mutableDescriptor() const noexcept
  {
    return descriptor_;
  }

  cusparseConstSpVecDescr_t constDescriptor() const noexcept
  {
    return const_descriptor_;
  }

private:
  cusparseSpVecDescr_t      descriptor_{nullptr};
  cusparseConstSpVecDescr_t const_descriptor_{nullptr};
};

class DenseVectorDescriptor
{
public:
  DenseVectorDescriptor(Index size, Real* vals)
  {
    checkCusparse(
        cusparseCreateDnVec(&descriptor_, size, vals, CUDA_R_64F),
        "cusparseCreateDnVec failed");
  }

  DenseVectorDescriptor(Index size, const Real* vals)
  {
    checkCusparse(
        cusparseCreateConstDnVec(&const_descriptor_,
                                 size,
                                 vals,
                                 CUDA_R_64F),
        "cusparseCreateConstDnVec failed");
  }

  ~DenseVectorDescriptor()
  {
    if (descriptor_ != nullptr)
    {
      cusparseDestroyDnVec(descriptor_);
    }
    if (const_descriptor_ != nullptr)
    {
      cusparseDestroyDnVec(const_descriptor_);
    }
  }

  cusparseDnVecDescr_t mutableDescriptor() const noexcept
  {
    return descriptor_;
  }

  cusparseConstDnVecDescr_t constDescriptor() const noexcept
  {
    return const_descriptor_;
  }

private:
  cusparseDnVecDescr_t      descriptor_{nullptr};
  cusparseConstDnVecDescr_t const_descriptor_{nullptr};
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

__global__ void axpbyKernel(Index       size,
                            Real        a,
                            const Real* x,
                            Real        b,
                            Real*       y)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    y[i] = a * x[i] + b * y[i];
  }
}
} // namespace

namespace detail
{
cublasHandle_t cublasHandle(void* stream)
{
  thread_local CublasHandle handle;
  return handle.get(stream);
}
} // namespace detail

void copy(DeviceConstVectorView src,
          DeviceVectorView      dst,
          CudaContext&          ctx)
{
  checkView(src.data(), src.size(), "Device copy has an invalid source view");
  checkView(dst.data(), dst.size(), "Device copy has an invalid destination view");
  require(src.size() == dst.size(),
          "Device view copy requires equal sizes");
  if (src.empty() || src.data() == dst.data())
  {
    return;
  }
  require(!detail::overlaps(src, dst),
          "Device view copy does not support partial overlap");

  device::copy(dst.data(),
               MemorySpace::Device,
               src.data(),
               MemorySpace::Device,
               static_cast<std::size_t>(src.size()) * sizeof(Real),
               ctx.stream());
}

void zero(DeviceVectorView vals, CudaContext& ctx)
{
  checkView(vals.data(), vals.size(), "zero has an invalid view");
  device::zero(vals.data(),
               static_cast<std::size_t>(vals.size()) * sizeof(Real),
               ctx.stream());
}

void axpby(Real                  a,
           DeviceConstVectorView x,
           Real                  b,
           DeviceVectorView      y,
           CudaContext&          ctx)
{
  checkView(x.data(), x.size(), "axpby has an invalid input view");
  checkView(y.data(), y.size(), "axpby has an invalid output view");
  require(x.size() == y.size(), "axpby requires equal vector sizes");
  if (x.empty())
  {
    return;
  }
  require(x.data() == y.data() || !detail::overlaps(x, y),
          "axpby does not support partial overlap");

  axpbyKernel<<<blocks(x.size()),
                kThreads,
                0,
                static_cast<cudaStream_t>(ctx.stream())>>>(
      x.size(), a, x.data(), b, y.data());
  device::checkLastError();
}

void gather(DeviceConstVectorView src,
            DeviceConstIndexView  indices,
            DeviceVectorView      dst,
            CudaContext&          ctx)
{
  checkView(src.data(), src.size(), "gather has an invalid source view");
  checkView(indices.data(), indices.size(), "gather has an invalid index view");
  checkView(dst.data(), dst.size(), "gather has an invalid output view");
  require(indices.size() == dst.size(), "gather output size mismatch");
  if (dst.empty())
  {
    return;
  }
  require(!detail::overlaps(src, dst), "gather does not support aliased vectors");

  DenseVectorDescriptor  dense(src.size(), src.data());
  SparseVectorDescriptor sparse(src.size(),
                                indices.size(),
                                indices.data(),
                                dst.data());
  checkCusparse(cusparseGather(cusparseHandle(ctx.stream()),
                               dense.constDescriptor(),
                               sparse.mutableDescriptor()),
                "cusparseGather failed");
}

void scatter(DeviceConstVectorView src,
             DeviceConstIndexView  indices,
             DeviceVectorView      dst,
             CudaContext&          ctx)
{
  checkView(src.data(), src.size(), "scatter has an invalid source view");
  checkView(indices.data(), indices.size(), "scatter has an invalid index view");
  checkView(dst.data(), dst.size(), "scatter has an invalid output view");
  require(src.size() == indices.size(),
          "scatter input size mismatch");
  if (src.empty())
  {
    return;
  }
  require(!detail::overlaps(src, dst),
          "scatter does not support aliased vectors");

  SparseVectorDescriptor sparse(dst.size(),
                                indices.size(),
                                indices.data(),
                                src.data());
  DenseVectorDescriptor  dense(dst.size(), dst.data());
  checkCusparse(cusparseScatter(cusparseHandle(ctx.stream()),
                                sparse.constDescriptor(),
                                dense.mutableDescriptor()),
                "cusparseScatter failed");
}

void dot(DeviceConstVectorView x,
         DeviceConstVectorView y,
         DeviceVectorView      out,
         CudaContext&          ctx)
{
  checkView(x.data(), x.size(), "dot has an invalid first input view");
  checkView(y.data(), y.size(), "dot has an invalid second input view");
  checkView(out.data(), out.size(), "dot has an invalid result view");
  require(x.size() == y.size() && out.size() == 1, "dot vector size mismatch");
  auto handle = detail::cublasHandle(ctx.stream());
  checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE),
              "cublasSetPointerMode failed");
  checkCublas(cublasDdot(handle,
                         x.size(),
                         x.data(),
                         1,
                         y.data(),
                         1,
                         out.data()),
              "cublasDdot failed");
}

} // namespace femx
