#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include <femx/linalg/handler/MatrixHandler.hpp>

namespace femx::linalg
{
namespace
{
struct HostCsrTransposeEntry
{
  std::uint64_t   source_layout{0};
  HostCsrPattern  pattern;
  HostIndexVector source_to_transpose;
};

struct HostCsrTransposeState
{
  std::vector<HostCsrTransposeEntry> entries;
};

HostCsrTransposeState& transposeState(std::shared_ptr<void>& storage)
{
  if (!storage)
  {
    storage = std::shared_ptr<void>(
        new HostCsrTransposeState,
        [](void* state)
        { delete static_cast<HostCsrTransposeState*>(state); });
  }
  return *static_cast<HostCsrTransposeState*>(storage.get());
}

HostCsrTransposeEntry makeTransposeEntry(const HostCsrPattern& src)
{
  HostIndexVector row_ptr(src.cols() + 1, 0);
  for (Index k = 0; k < src.nnz(); ++k)
  {
    ++row_ptr[src.colIndData()[k] + 1];
  }
  for (Index row = 0; row < src.cols(); ++row)
  {
    row_ptr[row + 1] += row_ptr[row];
  }

  HostIndexVector next = row_ptr;
  HostIndexVector col_ind(src.nnz());
  HostIndexVector source_to_transpose(src.nnz());
  for (Index row = 0; row < src.rows(); ++row)
  {
    for (Index k = src.rowPtrData()[row];
         k < src.rowPtrData()[row + 1];
         ++k)
    {
      const Index transpose_row   = src.colIndData()[k];
      const Index transpose_index = next[transpose_row]++;
      col_ind[transpose_index]    = row;
      source_to_transpose[k]      = transpose_index;
    }
  }

  return {src.layoutId(),
          HostCsrPattern(src.cols(),
                         src.rows(),
                         std::move(row_ptr),
                         std::move(col_ind)),
          std::move(source_to_transpose)};
}

HostCsrTransposeEntry& findOrCreateTransposeEntry(
    HostCsrTransposeState& state,
    const HostCsrPattern&  src)
{
  const auto iter = std::find_if(
      state.entries.begin(),
      state.entries.end(),
      [&src](const HostCsrTransposeEntry& entry)
      { return entry.source_layout == src.layoutId(); });
  if (iter != state.entries.end())
  {
    return *iter;
  }

  state.entries.push_back(makeTransposeEntry(src));
  return state.entries.back();
}

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

void MatrixHandler<HostCsrBackend>::transpose(
    const HostCsrMatrix& src,
    HostCsrMatrix&       dst) const
{
  require(&src != &dst, "CSR transpose does not support in-place output");
  auto& state = transposeState(transpose_state_);
  auto& entry = findOrCreateTransposeEntry(state, src.pattern());
  if (dst.pattern().layoutId() != entry.pattern.layoutId())
  {
    dst = HostCsrMatrix(entry.pattern);
  }
  for (Index k = 0; k < src.nnz(); ++k)
  {
    dst.valsData()[entry.source_to_transpose[k]] = src.valsData()[k];
  }
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

void MatrixHandler<CudaCsrBackend>::transpose(
    const DeviceCsrMatrix&,
    DeviceCsrMatrix&) const
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
