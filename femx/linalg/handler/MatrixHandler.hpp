#pragma once

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx::linalg
{

/** @brief Matrix operations associated with one execution backend. */
template <class Backend>
class MatrixHandler;

/** @brief Serial CPU sparse and dense matrix operations. */
template <>
class MatrixHandler<HostCsrBackend> final
{
public:
  explicit MatrixHandler(CpuContext& ctx) noexcept
    : ctx_(ctx), vec_handler_(ctx)
  {
  }

  void zero(HostCsrMatrix& mat) const;
  void zero(DenseMatrix& mat) const;
  void copy(const HostCsrMatrix& src, HostCsrMatrix& dst) const;

  void matvec(const HostCsrMatrix& mat,
              HostConstVectorView  x,
              HostVectorView       y,
              Real                 alpha = 1.0,
              Real                 beta  = 0.0) const;
  void matvecT(const HostCsrMatrix& mat,
               HostConstVectorView  x,
               HostVectorView       y,
               Real                 alpha = 1.0,
               Real                 beta  = 0.0) const;
  void matvec(const HostCsrMatrix& mat,
              HostConstVectorView  x,
              HostVector&          out) const;
  void matvecT(const HostCsrMatrix& mat,
               HostConstVectorView  x,
               HostVector&          out) const;

  void matvec(HostMatrixView<const Real> mat,
              HostConstVectorView        x,
              HostVectorView             y,
              Real                       alpha = 1.0,
              Real                       beta  = 0.0) const;
  void matvecT(HostMatrixView<const Real> mat,
               HostConstVectorView        x,
               HostVectorView             y,
               Real                       alpha = 1.0,
               Real                       beta  = 0.0) const;

  /** @brief Host CSR assembly needs no finalization step. */
  void finalize(HostCsrMatrix&) const noexcept
  {
  }

private:
  static void checkCopy(const HostCsrMatrix& src, const HostCsrMatrix& dst)
  {
    require(src.rows() == dst.rows() && src.cols() == dst.cols()
                && src.nnz() == dst.nnz()
                && src.pattern().layoutId() == dst.pattern().layoutId()
                && src.vals().size() == src.nnz()
                && dst.vals().size() == dst.nnz(),
            "CsrMatrix copy requires compatible source and destination graphs");
  }

  CpuContext&       ctx_;
  HostVectorHandler vec_handler_;
};

/** @brief CUDA sparse and dense matrix operations. */
template <>
class MatrixHandler<CudaCsrBackend> final
{
public:
  explicit MatrixHandler(CudaContext& ctx) noexcept
    : ctx_(ctx), vec_handler_(ctx)
  {
  }

  void zero(DeviceCsrMatrix& mat) const;
  void copy(const HostCsrMatrix& src, DeviceCsrMatrix& dst) const;
  void copy(const DeviceCsrMatrix& src, DeviceCsrMatrix& dst) const;
  void copy(const DeviceCsrMatrix& src, HostCsrMatrix& dst) const;

  void matvec(const DeviceCsrMatrix& mat,
              DeviceConstVectorView  x,
              DeviceVectorView       y,
              Real                   alpha = 1.0,
              Real                   beta  = 0.0) const;
  void matvecT(const DeviceCsrMatrix& mat,
               DeviceConstVectorView  x,
               DeviceVectorView       y,
               Real                   alpha = 1.0,
               Real                   beta  = 0.0) const;
  void matvec(const DeviceCsrMatrix& mat,
              DeviceConstVectorView  x,
              DeviceVector&          out) const;
  void matvecT(const DeviceCsrMatrix& mat,
               DeviceConstVectorView  x,
               DeviceVector&          out) const;

  void matvec(DeviceMatrixView<const Real> mat,
              DeviceConstVectorView        x,
              DeviceVectorView             y,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const;
  void matvecT(DeviceMatrixView<const Real> mat,
               DeviceConstVectorView        x,
               DeviceVectorView             y,
               Real                         alpha = 1.0,
               Real                         beta  = 0.0) const;

  /** @brief Device CSR assembly needs no finalization step. */
  void finalize(DeviceCsrMatrix&) const noexcept
  {
  }

private:
  template <MemorySpace SrcSpace, MemorySpace DstSpace>
  static void checkCopy(const CsrMatrix<SrcSpace>& src,
                        const CsrMatrix<DstSpace>& dst)
  {
    require(src.rows() == dst.rows() && src.cols() == dst.cols()
                && src.nnz() == dst.nnz()
                && src.pattern().layoutId() == dst.pattern().layoutId()
                && src.vals().size() == src.nnz()
                && dst.vals().size() == dst.nnz(),
            "CsrMatrix copy requires compatible source and destination graphs");
  }

  CudaContext&      ctx_;
  CudaVectorHandler vec_handler_;
};

using HostMatrixHandler = MatrixHandler<HostCsrBackend>;
using CudaMatrixHandler = MatrixHandler<CudaCsrBackend>;

} // namespace femx::linalg
