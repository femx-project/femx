#include <cuda_runtime.h>
#include <cusparse.h>

#include <cublas_v2.h>
#include <femx/linalg/handler/CudaHandles.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx::linalg
{
namespace
{
constexpr int kThreads = 256;

using detail::checkCublas;
using detail::checkCusparse;

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
    checkCusparse(cusparseCreateDnVec(&descriptor_, size, vals, CUDA_R_64F),
                  "cusparseCreateDnVec failed");
  }

  DenseVectorDescriptor(Index size, const Real* vals)
  {
    checkCusparse(cusparseCreateConstDnVec(&const_descriptor_,
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

void VectorHandler<CudaCsrBackend>::copy(DeviceConstVectorView src,
                                         DeviceVectorView      dst) const
{
  require(src.isValid(), "Device copy has an invalid source view");
  require(dst.isValid(), "Device copy has an invalid destination view");
  require(src.size() == dst.size(), "Device view copy requires equal sizes");
  if (src.empty() || src.data() == dst.data())
  {
    return;
  }
  require(!femx::detail::overlaps(src, dst),
          "Device view copy does not support partial overlap");
  cuda::copy(dst.data(),
             MemorySpace::Device,
             src.data(),
             MemorySpace::Device,
             static_cast<std::size_t>(src.size()) * sizeof(Real),
             ctx_.stream());
}

void VectorHandler<CudaCsrBackend>::copy(DeviceConstVectorView src,
                                         DeviceVector&         dst) const
{
  resize(dst, src.size());
  copy(src, dst.view());
}

void VectorHandler<CudaCsrBackend>::copy(HostConstVectorView src,
                                         DeviceVectorView    dst) const
{
  require(src.isValid(), "Host-to-Device copy has an invalid source view");
  require(dst.isValid(), "Host-to-Device copy has an invalid destination view");
  require(src.size() == dst.size(), "Host-to-Device view copy requires equal sizes");
  if (!src.empty())
  {
    cuda::copy(dst.data(),
               MemorySpace::Device,
               src.data(),
               MemorySpace::Host,
               static_cast<std::size_t>(src.size()) * sizeof(Real),
               ctx_.stream());
  }
}

void VectorHandler<CudaCsrBackend>::copy(HostConstVectorView src,
                                         DeviceVector&       dst) const
{
  resize(dst, src.size());
  copy(src, dst.view());
}

void VectorHandler<CudaCsrBackend>::copy(DeviceConstVectorView src,
                                         HostVectorView        dst) const
{
  require(src.isValid(), "Device-to-Host copy has an invalid source view");
  require(dst.isValid(), "Device-to-Host copy has an invalid destination view");
  require(src.size() == dst.size(), "Device-to-Host view copy requires equal sizes");
  if (!src.empty())
  {
    cuda::copy(dst.data(),
               MemorySpace::Host,
               src.data(),
               MemorySpace::Device,
               static_cast<std::size_t>(src.size()) * sizeof(Real),
               ctx_.stream());
  }
}

void VectorHandler<CudaCsrBackend>::copy(DeviceConstVectorView src,
                                         HostVector&           dst) const
{
  resize(dst, src.size());
  copy(src, dst.view());
}

void VectorHandler<CudaCsrBackend>::zero(DeviceVectorView vals) const
{
  require(vals.isValid(), "zero has an invalid view");
  if (!vals.empty())
  {
    cuda::zero(vals.data(),
               static_cast<std::size_t>(vals.size()) * sizeof(Real),
               ctx_.stream());
  }
}

void VectorHandler<CudaCsrBackend>::axpby(Real                  a,
                                          DeviceConstVectorView x,
                                          Real                  b,
                                          DeviceVectorView      y) const
{
  require(x.isValid(), "axpby has an invalid input view");
  require(y.isValid(), "axpby has an invalid output view");
  require(x.size() == y.size(), "axpby requires equal vector sizes");
  if (x.empty())
  {
    return;
  }
  require(x.data() == y.data() || !femx::detail::overlaps(x, y),
          "axpby does not support partial overlap");
  axpbyKernel<<<cuda::numBlocks(x.size(), kThreads),
                kThreads,
                0,
                static_cast<cudaStream_t>(ctx_.stream())>>>(
      x.size(), a, x.data(), b, y.data());
  cuda::checkLastError();
}

void VectorHandler<CudaCsrBackend>::gather(DeviceConstVectorView src,
                                           DeviceConstIndexView  indices,
                                           DeviceVectorView      dst) const
{
  require(src.isValid(), "gather has an invalid source view");
  require(indices.isValid(), "gather has an invalid index view");
  require(dst.isValid(), "gather has an invalid output view");
  require(indices.size() == dst.size(), "gather output size mismatch");
  if (dst.empty())
  {
    return;
  }
  require(!femx::detail::overlaps(src, dst),
          "gather does not support aliased vectors");
  DenseVectorDescriptor  dense(src.size(), src.data());
  SparseVectorDescriptor sparse(src.size(),
                                indices.size(),
                                indices.data(),
                                dst.data());
  checkCusparse(cusparseGather(detail::cusparseHandle(ctx_.stream()),
                               dense.constDescriptor(),
                               sparse.mutableDescriptor()),
                "cusparseGather failed");
}

void VectorHandler<CudaCsrBackend>::scatter(DeviceConstVectorView src,
                                            DeviceConstIndexView  indices,
                                            DeviceVectorView      dst) const
{
  require(src.isValid(), "scatter has an invalid source view");
  require(indices.isValid(), "scatter has an invalid index view");
  require(dst.isValid(), "scatter has an invalid output view");
  require(src.size() == indices.size(), "scatter input size mismatch");
  if (src.empty())
  {
    return;
  }
  require(!femx::detail::overlaps(src, dst),
          "scatter does not support aliased vectors");
  SparseVectorDescriptor sparse(dst.size(),
                                indices.size(),
                                indices.data(),
                                src.data());
  DenseVectorDescriptor  dense(dst.size(), dst.data());
  checkCusparse(cusparseScatter(detail::cusparseHandle(ctx_.stream()),
                                sparse.constDescriptor(),
                                dense.mutableDescriptor()),
                "cusparseScatter failed");
}

void VectorHandler<CudaCsrBackend>::dot(DeviceConstVectorView x,
                                        DeviceConstVectorView y,
                                        DeviceVectorView      out) const
{
  require(x.isValid(), "dot has an invalid first input view");
  require(y.isValid(), "dot has an invalid second input view");
  require(out.isValid(), "dot has an invalid result view");
  require(x.size() == y.size() && out.size() == 1,
          "dot vector size mismatch");
  auto handle = detail::cublasHandle(ctx_.stream());
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

} // namespace femx::linalg
