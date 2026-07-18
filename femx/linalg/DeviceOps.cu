#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include <femx/linalg/CsrTranspose.hpp>

namespace femx
{
namespace
{
constexpr int BLOCK_SIZE = 256;

template <class T>
void checkView(const T* data, Index size, const char* name)
{
  if (size < 0 || (size > 0 && data == nullptr))
  {
    throw std::runtime_error(name);
  }
}

template <class T, class U>
bool overlaps(const T* lhs, Index lhs_size, const U* rhs, Index rhs_size)
{
  if (lhs == nullptr || rhs == nullptr || lhs_size <= 0 || rhs_size <= 0)
  {
    return false;
  }

  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
  const auto lhs_end   = lhs_begin + static_cast<std::size_t>(lhs_size) * sizeof(T);
  const auto rhs_end   = rhs_begin + static_cast<std::size_t>(rhs_size) * sizeof(U);
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

unsigned int blocks(Index size)
{
  return static_cast<unsigned int>(
      (static_cast<std::int64_t>(size) + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

__global__ void axpbyKernel(Index       size,
                            Real        a,
                            const Real* x,
                            Real        b,
                            Real*       y)
{
  const Index i = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    y[i] = a * x[i] + b * y[i];
  }
}

__global__ void csrApplyKernel(Index        rows,
                               const Index* row_ptr,
                               const Index* col_ind,
                               const Real*  vals,
                               const Real*  x,
                               Real*        y)
{
  const Index row =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows)
  {
    return;
  }

  Real val = 0.0;
  for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
  {
    val += vals[k] * x[col_ind[k]];
  }
  y[row] = val;
}

__global__ void trValsKernel(Index        size,
                             const Real*  src,
                             const Index* src_to_tr,
                             Real*        dst)
{
  const Index k = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < size)
  {
    dst[src_to_tr[k]] = src[k];
  }
}
} // namespace

void copy(DeviceConstVectorView src,
          DeviceVectorView      dst,
          CudaContext&          ctx)
{
  checkView(src.data(), src.size(), "Device copy has an invalid source view");
  checkView(dst.data(), dst.size(), "Device copy has an invalid destination view");
  if (src.size() != dst.size())
  {
    throw std::runtime_error("Device view copy requires equal sizes");
  }
  if (src.empty() || src.data() == dst.data())
  {
    return;
  }
  if (overlaps(src.data(), src.size(), dst.data(), dst.size()))
  {
    throw std::runtime_error("Device view copy does not support partial overlap");
  }

  device::copy(dst.data(),
               MemorySpace::Device,
               src.data(),
               MemorySpace::Device,
               static_cast<std::size_t>(src.size()) * sizeof(Real),
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
  if (x.size() != y.size())
  {
    throw std::runtime_error("axpby requires equal vector sizes");
  }
  if (x.empty())
  {
    return;
  }
  if (x.data() != y.data()
      && overlaps(x.data(), x.size(), y.data(), y.size()))
  {
    throw std::runtime_error("axpby does not support partial overlap");
  }

  axpbyKernel<<<blocks(x.size()), BLOCK_SIZE, 0, static_cast<cudaStream_t>(ctx.stream())>>>(
      x.size(), a, x.data(), b, y.data());
  device::checkLastError();
}

void apply(const DeviceCsrMatrix& mat,
           DeviceConstVectorView  x,
           DeviceVectorView       y,
           CudaContext&           ctx)
{
  checkView(x.data(), x.size(), "CSR apply has an invalid input view");
  checkView(y.data(), y.size(), "CSR apply has an invalid output view");
  if (x.size() != mat.cols() || y.size() != mat.rows())
  {
    throw std::runtime_error("CSR apply vector size mismatch");
  }
  if (mat.rows() > 0 && mat.rowPtrData() == nullptr)
  {
    throw std::runtime_error("CSR apply has no Device row offsets");
  }
  if (mat.nnz() > 0
      && (mat.colIndData() == nullptr || mat.valsData() == nullptr))
  {
    throw std::runtime_error("CSR apply has incomplete Device storage");
  }
  if (overlaps(x.data(), x.size(), y.data(), y.size()))
  {
    throw std::runtime_error("CSR apply does not support in-place vectors");
  }
  if (overlaps(mat.valsData(), mat.nnz(), y.data(), y.size()))
  {
    throw std::runtime_error("CSR apply output aliases matrix values");
  }
  if (mat.rows() == 0)
  {
    return;
  }

  csrApplyKernel<<<blocks(mat.rows()), BLOCK_SIZE, 0, static_cast<cudaStream_t>(ctx.stream())>>>(
      mat.rows(),
      mat.rowPtrData(),
      mat.colIndData(),
      mat.valsData(),
      x.data(),
      y.data());
  device::checkLastError();
}

void trVals(const DeviceCsrMatrix&       src,
            const DeviceCsrTransposeMap& map,
            DeviceCsrMatrix&             dst,
            CudaContext&                 ctx)
{
  if (src.graph().layoutId() != map.srcGraph().layoutId()
      || dst.graph().layoutId() != map.trGraph().layoutId()
      || src.nnz() != map.srcToTr().size() || src.nnz() != dst.nnz())
  {
    throw std::runtime_error("Device CSR transpose graph mismatch");
  }
  if (src.nnz() == 0)
  {
    return;
  }
  if (src.valsData() == nullptr || dst.valsData() == nullptr
      || map.srcToTr().data() == nullptr)
  {
    throw std::runtime_error("Device CSR transpose has incomplete storage");
  }
  if (overlaps(src.valsData(), src.nnz(), dst.valsData(), dst.nnz()))
  {
    throw std::runtime_error("Device CSR transpose does not support in-place values");
  }

  trValsKernel<<<blocks(src.nnz()), BLOCK_SIZE, 0, static_cast<cudaStream_t>(ctx.stream())>>>(
      src.nnz(), src.valsData(), map.srcToTr().data(), dst.valsData());
  device::checkLastError();
}

} // namespace femx
