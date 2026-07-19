#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#if defined(FEMX_HAS_CUDA)
#include <cuda_runtime_api.h>

#include <string>

#include <cublas_v2.h>
#endif

#include <femx/common/Checks.hpp>
#include <femx/linalg/Dense.hpp>

namespace femx
{
#if defined(FEMX_HAS_CUDA)
namespace detail
{
cublasHandle_t cublasHandle(void* stream);
} // namespace detail
#endif

namespace
{
void checkDenseApply(Index               rows,
                     Index               cols,
                     const Real*         vals,
                     HostConstVectorView x,
                     HostVectorView      y,
                     bool                transpose)
{
  const Index in_size  = transpose ? rows : cols;
  const Index out_size = transpose ? cols : rows;
  require(rows >= 0 && cols >= 0 && x.size() == in_size
              && y.size() == out_size
              && (rows * cols == 0 || vals != nullptr),
          "Dense apply received incompatible storage");
  require(!detail::overlaps(x, y)
              && !detail::overlaps(vals, rows * cols, y.data(), y.size()),
          "Dense apply does not support aliased vectors");
}

void checkDeviceDenseApply(DeviceMatrixView<const Real> mat,
                           DeviceConstVectorView        x,
                           DeviceVectorView             y,
                           bool                         transpose)
{
  const Index in_size  = transpose ? mat.rows() : mat.cols();
  const Index out_size = transpose ? mat.cols() : mat.rows();
  require(mat.rows() >= 0 && mat.cols() >= 0 && x.size() == in_size
              && y.size() == out_size
              && (mat.rows() * mat.cols() == 0 || mat.data() != nullptr),
          "Device dense apply received incompatible storage");
  require(!detail::overlaps(x, y)
              && !detail::overlaps(
                  mat.data(), mat.rows() * mat.cols(), y.data(), y.size()),
          "Device dense apply does not support aliased vectors");
}

void scaleDevice(DeviceVectorView vals, Real scale, CudaContext& ctx)
{
  if (vals.empty() || scale == 1.0)
  {
    return;
  }
  axpby(0.0,
        DeviceConstVectorView(vals.data(), vals.size()),
        scale,
        vals,
        ctx);
}

#if defined(FEMX_HAS_CUDA)
void checkCublas(cublasStatus_t status, const char* op)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    throw std::runtime_error(std::string(op) + ": "
                             + cublasGetStatusString(status));
  }
}
#endif
} // namespace

DenseMatrix::DenseMatrix(Index rows, Index cols)
  : rows_(rows), cols_(cols), vals_(rows * cols, Real{})
{
}

void DenseMatrix::resize(Index rows, Index cols)
{
  rows_ = rows;
  cols_ = cols;
  vals_.assign(rows_ * cols_, Real{});
}

void DenseMatrix::setZero()
{
  vals_.setZero();
}

Index DenseMatrix::rows() const
{
  return rows_;
}

Index DenseMatrix::cols() const
{
  return cols_;
}

Index DenseMatrix::size() const
{
  return vals_.size();
}

Real& DenseMatrix::operator()(Index i, Index j)
{
  return vals_[i * cols_ + j];
}

Real DenseMatrix::operator()(Index i, Index j) const
{
  return vals_[i * cols_ + j];
}

Real* DenseMatrix::data()
{
  return vals_.data();
}

const Real* DenseMatrix::data() const
{
  return vals_.data();
}

HostMatrixView<Real> DenseMatrix::view()
{
  return {data(), rows_, cols_};
}

HostMatrixView<const Real> DenseMatrix::view() const
{
  return {data(), rows_, cols_};
}

void apply(HostMatrixView<const Real> mat,
           HostConstVectorView        x,
           HostVectorView             y,
           CpuContext&,
           Real alpha,
           Real beta)
{
  checkDenseApply(mat.rows(), mat.cols(), mat.data(), x, y, false);
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

void applyT(HostMatrixView<const Real> mat,
            HostConstVectorView        x,
            HostVectorView             y,
            CpuContext&,
            Real alpha,
            Real beta)
{
  checkDenseApply(mat.rows(), mat.cols(), mat.data(), x, y, true);
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

void apply(DeviceMatrixView<const Real> mat,
           DeviceConstVectorView        x,
           DeviceVectorView             y,
           CudaContext&                 ctx,
           Real                         alpha,
           Real                         beta)
{
#if defined(FEMX_HAS_CUDA)
  checkDeviceDenseApply(mat, x, y, false);
  if (mat.rows() == 0)
  {
    return;
  }
  if (mat.cols() == 0)
  {
    scaleDevice(y, beta, ctx);
    return;
  }
  auto handle = detail::cublasHandle(ctx.stream());
  checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
              "cublasSetPointerMode failed");
  checkCublas(cublasDgemv(handle,
                          CUBLAS_OP_T,
                          mat.cols(),
                          mat.rows(),
                          &alpha,
                          mat.data(),
                          mat.cols(),
                          x.data(),
                          1,
                          &beta,
                          y.data(),
                          1),
              "cublasDgemv failed");
#else
  (void) mat;
  (void) x;
  (void) y;
  (void) ctx;
  (void) alpha;
  (void) beta;
  throw std::runtime_error("Device dense apply requires CUDA");
#endif
}

void applyT(DeviceMatrixView<const Real> mat,
            DeviceConstVectorView        x,
            DeviceVectorView             y,
            CudaContext&                 ctx,
            Real                         alpha,
            Real                         beta)
{
#if defined(FEMX_HAS_CUDA)
  checkDeviceDenseApply(mat, x, y, true);
  if (mat.cols() == 0)
  {
    return;
  }
  if (mat.rows() == 0)
  {
    scaleDevice(y, beta, ctx);
    return;
  }
  auto handle = detail::cublasHandle(ctx.stream());
  checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
              "cublasSetPointerMode failed");
  checkCublas(cublasDgemv(handle,
                          CUBLAS_OP_N,
                          mat.cols(),
                          mat.rows(),
                          &alpha,
                          mat.data(),
                          mat.cols(),
                          x.data(),
                          1,
                          &beta,
                          y.data(),
                          1),
              "cublasDgemv(transpose) failed");
#else
  (void) mat;
  (void) x;
  (void) y;
  (void) ctx;
  (void) alpha;
  (void) beta;
  throw std::runtime_error("Device dense transpose requires CUDA");
#endif
}

namespace linalg
{

DenseLinearSolver::DenseLinearSolver(Real pivot_tolerance)
  : pivot_tolerance_(pivot_tolerance)
{
}

void DenseLinearSolver::solve(const HostCsrMatrix& mat,
                              const HostVector&    rhs,
                              HostVector&          out,
                              CpuContext&)
{
  require(mat.rows() == mat.cols() && rhs.size() == mat.rows(),
          "DenseLinearSolver received inconsistent CSR dimensions");
  HostVector dense;
  sample(mat, false, dense);
  solveDense(std::move(dense), rhs, out, mat.cols());
}

void DenseLinearSolver::solveT(const HostCsrMatrix& mat,
                               const HostVector&    rhs,
                               HostVector&          out,
                               CpuContext&)
{
  require(mat.rows() == mat.cols() && rhs.size() == mat.cols(),
          "DenseLinearSolver received inconsistent transposed CSR dimensions");
  HostVector dense;
  sample(mat, true, dense);
  solveDense(std::move(dense), rhs, out, mat.rows());
}

void DenseLinearSolver::sample(const HostCsrMatrix& mat,
                               bool                 tr,
                               HostVector&          dense) const
{
  const Index size = mat.rows();
  dense.assign(size * size, 0.0);
  for (Index row = 0; row < size; ++row)
  {
    for (Index k = mat.rowPtrData()[row]; k < mat.rowPtrData()[row + 1]; ++k)
    {
      const Index col          = mat.colIndData()[k];
      const Index i            = tr ? col : row;
      const Index j            = tr ? row : col;
      dense[entry(i, j, size)] = mat.valsData()[k];
    }
  }
}

void DenseLinearSolver::solveDense(HostVector        mat,
                                   const HostVector& rhs,
                                   HostVector&       out,
                                   Index             size) const
{
  HostVector b(size, 0.0);
  for (Index i = 0; i < size; ++i)
  {
    b[i] = rhs[i];
  }

  for (Index k = 0; k < size; ++k)
  {
    Index pivot = k;
    Real  best  = std::abs(mat[entry(k, k, size)]);
    for (Index i = k + 1; i < size; ++i)
    {
      const Real candidate = std::abs(mat[entry(i, k, size)]);
      if (candidate > best)
      {
        best  = candidate;
        pivot = i;
      }
    }
    if (best <= pivot_tolerance_)
    {
      throw std::runtime_error(
          "DenseLinearSolver detected singular matrix");
    }
    if (pivot != k)
    {
      for (Index j = k; j < size; ++j)
      {
        std::swap(mat[entry(k, j, size)], mat[entry(pivot, j, size)]);
      }
      std::swap(b[k], b[pivot]);
    }
    for (Index i = k + 1; i < size; ++i)
    {
      const Real factor      = mat[entry(i, k, size)] / mat[entry(k, k, size)];
      mat[entry(i, k, size)] = 0.0;
      for (Index j = k + 1; j < size; ++j)
      {
        mat[entry(i, j, size)] -= factor * mat[entry(k, j, size)];
      }
      b[i] -= factor * b[k];
    }
  }

  resizeOrZero(out, size);
  for (Index i = size; i-- > 0;)
  {
    Real sum = b[i];
    for (Index j = i + 1; j < size; ++j)
    {
      sum -= mat[entry(i, j, size)] * out[j];
    }
    out[i] = sum / mat[entry(i, i, size)];
  }
}

Index DenseLinearSolver::entry(Index row, Index col, Index size)
{
  return row * size + col;
}

} // namespace linalg
} // namespace femx
