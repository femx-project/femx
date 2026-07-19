#include <stdexcept>

#include <femx/common/Checks.hpp>
#include <femx/linalg/CsrMatrix.hpp>

namespace femx
{
namespace
{
void checkApply(const HostCsrMatrix& mat,
                HostConstVectorView  x,
                HostVectorView       y,
                bool                 transpose)
{
  const Index in_size  = transpose ? mat.rows() : mat.cols();
  const Index out_size = transpose ? mat.cols() : mat.rows();
  require(x.size() == in_size && y.size() == out_size,
          "Host CSR apply vector size mismatch");
  require(!detail::overlaps(x, y),
          "Host CSR apply does not support in-place views");
}
} // namespace

void apply(const HostCsrMatrix& mat,
           HostConstVectorView  x,
           HostVectorView       y,
           CpuContext&,
           Real alpha,
           Real beta)
{
  checkApply(mat, x, y, false);
  for (Index row = 0; row < mat.rows(); ++row)
  {
    Real val = 0.0;
    for (Index k = mat.rowPtrData()[row];
         k < mat.rowPtrData()[row + 1];
         ++k)
    {
      val += mat.valsData()[k] * x[mat.colIndData()[k]];
    }
    y[row] = alpha * val + beta * y[row];
  }
}

void applyT(const HostCsrMatrix& mat,
            HostConstVectorView  x,
            HostVectorView       y,
            CpuContext&,
            Real alpha,
            Real beta)
{
  checkApply(mat, x, y, true);
  for (Index col = 0; col < mat.cols(); ++col)
  {
    y[col] *= beta;
  }
  for (Index row = 0; row < mat.rows(); ++row)
  {
    const Real val = alpha * x[row];
    for (Index k = mat.rowPtrData()[row];
         k < mat.rowPtrData()[row + 1];
         ++k)
    {
      y[mat.colIndData()[k]] += mat.valsData()[k] * val;
    }
  }
}

#if !defined(FEMX_HAS_CUDA)
namespace
{
[[noreturn]] void cudaUnavailable()
{
  throw std::runtime_error(
      "femx was built without the CUDA execution backend");
}
} // namespace

void apply(const DeviceCsrMatrix&,
           DeviceConstVectorView,
           DeviceVectorView,
           CudaContext&,
           Real,
           Real)
{
  cudaUnavailable();
}

void applyT(const DeviceCsrMatrix&,
            DeviceConstVectorView,
            DeviceVectorView,
            CudaContext&,
            Real,
            Real)
{
  cudaUnavailable();
}
#endif

} // namespace femx
