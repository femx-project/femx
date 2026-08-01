#include <cuda_runtime.h>
#include <cusparse.h>

#include <cublas_v2.h>
#include <femx/common/Cuda.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaHandles.hpp>
#include <femx/linalg/cuda/CudaVectorHandler.hpp>

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
                         const Index* idx,
                         Real*        vals)
  {
    checkCusparse(cusparseCreateSpVec(&descriptor_,
                                      size,
                                      nnz,
                                      const_cast<Index*>(idx),
                                      vals,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_64F),
                  "cusparseCreateSpVec failed");
  }

  SparseVectorDescriptor(Index        size,
                         Index        nnz,
                         const Index* idx,
                         const Real*  vals)
  {
    checkCusparse(cusparseCreateConstSpVec(&const_descriptor_,
                                           size,
                                           nnz,
                                           idx,
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

template <class T>
void assignVector(DeviceVector<T>& out,
                  Index            size,
                  T                val,
                  CudaContext&     ctx)
{
  require(size >= 0, "Vector size must be non-negative");
  if (out.size() != size)
  {
    out.assign(size, val);
    return;
  }
  if (out.empty())
  {
    return;
  }
  device::fill(out.data(), out.size(), val, ctx.stream());
}
} // namespace

void CudaVectorHandler::assign(DeviceVector<Real>& out,
                               Index               size,
                               Real                val) const
{
  assignVector(out, size, val, ctx_);
}

void CudaVectorHandler::assign(DeviceVector<Index>& out,
                               Index                size,
                               Index                val) const
{
  assignVector(out, size, val, ctx_);
}

void CudaVectorHandler::copy(DeviceVectorView<const Real> src,
                             DeviceVectorView<Real>       dst) const
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

  device::copy(dst.data(),
               MemorySpace::Device,
               src.data(),
               MemorySpace::Device,
               static_cast<std::size_t>(src.size()) * sizeof(Real),
               ctx_.stream());
}

void CudaVectorHandler::copy(DeviceVectorView<const Real> src,
                             DeviceVector<Real>&          dst) const
{
  resize(dst, src.size());
  copy(src, dst.view());
}

void CudaVectorHandler::copy(HostVectorView<const Real> src,
                             DeviceVectorView<Real>     dst) const
{
  require(src.isValid(), "Host-to-Device copy has an invalid source view");
  require(dst.isValid(), "Host-to-Device copy has an invalid destination view");
  require(src.size() == dst.size(), "Host-to-Device view copy requires equal sizes");

  if (!src.empty())
  {
    device::copy(dst.data(),
                 MemorySpace::Device,
                 src.data(),
                 MemorySpace::Host,
                 static_cast<std::size_t>(src.size()) * sizeof(Real),
                 ctx_.stream());
  }
}

void CudaVectorHandler::copy(HostVectorView<const Real> src,
                             DeviceVector<Real>&        dst) const
{
  resize(dst, src.size());
  copy(src, dst.view());
}

void CudaVectorHandler::copy(DeviceVectorView<const Real> src,
                             HostVectorView<Real>         dst) const
{
  require(src.isValid(), "Device-to-Host copy has an invalid source view");
  require(dst.isValid(), "Device-to-Host copy has an invalid destination view");
  require(src.size() == dst.size(), "Device-to-Host view copy requires equal sizes");
  if (!src.empty())
  {
    device::copy(dst.data(),
                 MemorySpace::Host,
                 src.data(),
                 MemorySpace::Device,
                 static_cast<std::size_t>(src.size()) * sizeof(Real),
                 ctx_.stream());
  }
}

void CudaVectorHandler::copy(DeviceVectorView<const Real> src,
                             HostVector<Real>&            dst) const
{
  resize(dst, src.size());
  copy(src, dst.view());
}

void CudaVectorHandler::zero(DeviceVectorView<Real> vals) const
{
  require(vals.isValid(), "zero has an invalid view");
  if (!vals.empty())
  {
    device::zero(vals.data(),
                 static_cast<std::size_t>(vals.size()) * sizeof(Real),
                 ctx_.stream());
  }
}

void CudaVectorHandler::axpby(Real                         a,
                              DeviceVectorView<const Real> x,
                              Real                         b,
                              DeviceVectorView<Real>       y) const
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

void CudaVectorHandler::gather(DeviceVectorView<const Real>  src,
                               DeviceVectorView<const Index> idx,
                               DeviceVectorView<Real>        dst) const
{
  require(src.isValid(), "gather has an invalid source view");
  require(idx.isValid(), "gather has an invalid index view");
  require(dst.isValid(), "gather has an invalid output view");
  require(idx.size() == dst.size(), "gather output size mismatch");

  if (dst.empty())
  {
    return;
  }

  require(!femx::detail::overlaps(src, dst),
          "gather does not support aliased vectors");

  DenseVectorDescriptor  dense(src.size(), src.data());
  SparseVectorDescriptor sparse(src.size(),
                                idx.size(),
                                idx.data(),
                                dst.data());

  checkCusparse(cusparseGather(detail::cusparseHandle(ctx_),
                               dense.constDescriptor(),
                               sparse.mutableDescriptor()),
                "cusparseGather failed");
}

void CudaVectorHandler::scatter(DeviceVectorView<const Real>  src,
                                DeviceVectorView<const Index> idx,
                                DeviceVectorView<Real>        dst) const
{
  require(src.isValid(), "scatter has an invalid source view");
  require(idx.isValid(), "scatter has an invalid index view");
  require(dst.isValid(), "scatter has an invalid output view");
  require(src.size() == idx.size(), "scatter input size mismatch");

  if (src.empty())
  {
    return;
  }

  require(!femx::detail::overlaps(src, dst),
          "scatter does not support aliased vectors");

  SparseVectorDescriptor sparse(dst.size(),
                                idx.size(),
                                idx.data(),
                                src.data());
  DenseVectorDescriptor  dense(dst.size(), dst.data());
  checkCusparse(cusparseScatter(detail::cusparseHandle(ctx_),
                                sparse.constDescriptor(),
                                dense.mutableDescriptor()),
                "cusparseScatter failed");
}

void CudaVectorHandler::dot(DeviceVectorView<const Real> x,
                            DeviceVectorView<const Real> y,
                            DeviceVectorView<Real>       out) const
{
  require(x.isValid(), "dot has an invalid first input view");
  require(y.isValid(), "dot has an invalid second input view");
  require(out.isValid(), "dot has an invalid result view");
  require(x.size() == y.size() && out.size() == 1,
          "dot vector size mismatch");

  auto handle = detail::cublasHandle(ctx_);
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
