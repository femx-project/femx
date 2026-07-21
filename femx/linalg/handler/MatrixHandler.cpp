#include <stdexcept>

#include <femx/linalg/handler/MatrixHandler.hpp>

namespace femx::linalg
{
namespace
{
void checkCsrMatvec(const HostCsrMatrix& mat,
                    HostConstVectorView  x,
                    HostVectorView       y,
                    bool                 transpose)
{
  const Index in_size  = transpose ? mat.rows() : mat.cols();
  const Index out_size = transpose ? mat.cols() : mat.rows();
  require(x.size() == in_size && y.size() == out_size,
          "Host CSR matvec vector size mismatch");
  require(!femx::detail::overlaps(x, y),
          "Host CSR matvec does not support in-place views");
}

void checkDenseMatvec(HostMatrixView<const Real> mat,
                      HostConstVectorView        x,
                      HostVectorView             y,
                      bool                       transpose)
{
  const Index in_size  = transpose ? mat.rows() : mat.cols();
  const Index out_size = transpose ? mat.cols() : mat.rows();
  require(mat.rows() >= 0 && mat.cols() >= 0 && x.size() == in_size
              && y.size() == out_size
              && (mat.rows() * mat.cols() == 0 || mat.data() != nullptr),
          "Dense matvec received incompatible storage");
  require(!femx::detail::overlaps(x, y)
              && !femx::detail::overlaps(mat.data(),
                                         mat.rows() * mat.cols(),
                                         y.data(),
                                         y.size()),
          "Dense matvec does not support aliased vectors");
}
} // namespace

void MatrixHandler<HostCsrBackend>::zero(HostCsrMatrix& mat) const
{
  vec_handler_.zero(mat.vals().view());
}

void MatrixHandler<HostCsrBackend>::zero(DenseMatrix& mat) const
{
  vec_handler_.zero(mat.vals().view());
}

void MatrixHandler<HostCsrBackend>::copy(const HostCsrMatrix& src,
                                         HostCsrMatrix&       dst) const
{
  checkCopy(src, dst);
  vec_handler_.copy(src.vals().view(), dst.vals().view());
}

void MatrixHandler<HostCsrBackend>::matvec(const HostCsrMatrix& mat,
                                           HostConstVectorView  x,
                                           HostVectorView       y,
                                           Real                 alpha,
                                           Real                 beta) const
{
  checkCsrMatvec(mat, x, y, false);
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

void MatrixHandler<HostCsrBackend>::matvecT(const HostCsrMatrix& mat,
                                            HostConstVectorView  x,
                                            HostVectorView       y,
                                            Real                 alpha,
                                            Real                 beta) const
{
  checkCsrMatvec(mat, x, y, true);
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

void MatrixHandler<HostCsrBackend>::matvec(const HostCsrMatrix& mat,
                                           HostConstVectorView  x,
                                           HostVector&          out) const
{
  if (out.size() != mat.rows())
  {
    out.resize(mat.rows());
  }
  matvec(mat, x, out.view());
}

void MatrixHandler<HostCsrBackend>::matvecT(const HostCsrMatrix& mat,
                                            HostConstVectorView  x,
                                            HostVector&          out) const
{
  if (out.size() != mat.cols())
  {
    out.resize(mat.cols());
  }
  matvecT(mat, x, out.view());
}

void MatrixHandler<HostCsrBackend>::matvec(HostMatrixView<const Real> mat,
                                           HostConstVectorView        x,
                                           HostVectorView             y,
                                           Real                       alpha,
                                           Real                       beta) const
{
  checkDenseMatvec(mat, x, y, false);
  for (Index row = 0; row < mat.rows(); ++row)
  {
    Real sum = 0.0;
    for (Index col = 0; col < mat.cols(); ++col)
    {
      sum += mat(row, col) * x[col];
    }
    y[row] = alpha * sum + beta * y[row];
  }
}

void MatrixHandler<HostCsrBackend>::matvecT(HostMatrixView<const Real> mat,
                                            HostConstVectorView        x,
                                            HostVectorView             y,
                                            Real                       alpha,
                                            Real                       beta) const
{
  checkDenseMatvec(mat, x, y, true);
  for (Index col = 0; col < mat.cols(); ++col)
  {
    Real sum = 0.0;
    for (Index row = 0; row < mat.rows(); ++row)
    {
      sum += mat(row, col) * x[row];
    }
    y[col] = alpha * sum + beta * y[col];
  }
}

void MatrixHandler<CudaCsrBackend>::zero(DeviceCsrMatrix& mat) const
{
  vec_handler_.zero(mat.vals().view());
}

void MatrixHandler<CudaCsrBackend>::copy(const HostCsrMatrix& src,
                                         DeviceCsrMatrix&     dst) const
{
  checkCopy(src, dst);
  vec_handler_.copy(src.vals(), dst.vals());
}

void MatrixHandler<CudaCsrBackend>::copy(const DeviceCsrMatrix& src,
                                         DeviceCsrMatrix&       dst) const
{
  checkCopy(src, dst);
  vec_handler_.copy(src.vals(), dst.vals());
}

void MatrixHandler<CudaCsrBackend>::copy(const DeviceCsrMatrix& src,
                                         HostCsrMatrix&         dst) const
{
  checkCopy(src, dst);
  vec_handler_.copy(src.vals(), dst.vals());
}

void MatrixHandler<CudaCsrBackend>::matvec(const DeviceCsrMatrix& mat,
                                           DeviceConstVectorView  x,
                                           DeviceVector&          out) const
{
  if (out.size() != mat.rows())
  {
    out.resize(mat.rows());
  }
  matvec(mat, x, out.view());
}

void MatrixHandler<CudaCsrBackend>::matvecT(const DeviceCsrMatrix& mat,
                                            DeviceConstVectorView  x,
                                            DeviceVector&          out) const
{
  if (out.size() != mat.cols())
  {
    out.resize(mat.cols());
  }
  matvecT(mat, x, out.view());
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

void MatrixHandler<CudaCsrBackend>::matvec(const DeviceCsrMatrix&,
                                           DeviceConstVectorView,
                                           DeviceVectorView,
                                           Real,
                                           Real) const
{
  cudaUnavailable();
}

void MatrixHandler<CudaCsrBackend>::matvecT(const DeviceCsrMatrix&,
                                            DeviceConstVectorView,
                                            DeviceVectorView,
                                            Real,
                                            Real) const
{
  cudaUnavailable();
}

void MatrixHandler<CudaCsrBackend>::matvec(DeviceMatrixView<const Real>,
                                           DeviceConstVectorView,
                                           DeviceVectorView,
                                           Real,
                                           Real) const
{
  cudaUnavailable();
}

void MatrixHandler<CudaCsrBackend>::matvecT(DeviceMatrixView<const Real>,
                                            DeviceConstVectorView,
                                            DeviceVectorView,
                                            Real,
                                            Real) const
{
  cudaUnavailable();
}
#endif

} // namespace femx::linalg
