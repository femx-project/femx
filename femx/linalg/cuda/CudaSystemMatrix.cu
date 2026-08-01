#include <cuda_runtime.h>

#include <femx/common/Cuda.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>

namespace femx::linalg
{
namespace
{
constexpr int kThreads = 256;

__global__ void markConstraintsKernel(Index        count,
                                      const Index* rows,
                                      Index        mat_rows,
                                      Index*       row_to_constraint)
{
  const Index ib =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ib < count)
  {
    const Index row = rows[ib];
    if (row >= 0 && row < mat_rows)
    {
      row_to_constraint[row] = ib;
    }
  }
}

__global__ void replaceConstraintRowsKernel(
    Index        count,
    const Index* rows,
    const Index* row_offsets,
    const Index* col_inds,
    Real*        mat_vals,
    Real         diag,
    Real*        rhs,
    const Real*  vals)
{
  const Index ib = static_cast<Index>(blockIdx.x);
  if (ib >= count)
  {
    return;
  }

  const Index row = rows[ib];
  for (Index entry = row_offsets[row] + threadIdx.x;
       entry < row_offsets[row + 1];
       entry += blockDim.x)
  {
    mat_vals[entry] = col_inds[entry] == row ? diag : 0.0;
  }
  if (threadIdx.x == 0 && rhs != nullptr)
  {
    rhs[row] = vals[ib];
  }
}

__global__ void eliminateConstraintColumnsKernel(
    Index        mat_rows,
    const Index* row_offsets,
    const Index* col_inds,
    const Index* row_to_constraint,
    const Real*  vals,
    Real*        mat_vals,
    Real*        rhs)
{
  const Index row = static_cast<Index>(blockIdx.x);
  if (row >= mat_rows)
  {
    return;
  }

  for (Index entry = row_offsets[row] + threadIdx.x;
       entry < row_offsets[row + 1];
       entry += blockDim.x)
  {
    const Index ib = row_to_constraint[col_inds[entry]];
    if (ib >= 0)
    {
      const Real val = mat_vals[entry];
      if (row_to_constraint[row] < 0)
      {
        atomicAdd(rhs + row, -val * vals[ib]);
      }
      mat_vals[entry] = 0.0;
    }
  }
}

} // namespace

void CudaSystemMatrix::ensureConstraints(
    DeviceVectorView<const Index> rows)
{
  require(rows.isValid(), "CUDA constrained-row view is invalid");
  if (constraints_.layout_id == mat_.pattern().layoutId()
      && constraints_.rows == rows.data()
      && constraints_.count == rows.size())
  {
    return;
  }

  constraints_.layout_id = mat_.pattern().layoutId();
  constraints_.rows      = rows.data();
  constraints_.count     = rows.size();
  ctx_.vectorHandler().assign(constraints_.row_to_constraint,
                              mat_.rows(),
                              -1);
  if (!rows.empty())
  {
    markConstraintsKernel<<<cuda::numBlocks(rows.size(), kThreads),
                            kThreads,
                            0,
                            static_cast<cudaStream_t>(ctx_.stream())>>>(
        rows.size(),
        rows.data(),
        mat_.rows(),
        constraints_.row_to_constraint.data());
    cuda::checkLastError();
  }
}

void CudaSystemMatrix::replaceRows(DeviceVectorView<const Index> rows,
                                   Real                          diag)
{
  ensureConstraints(rows);
  if (rows.empty())
  {
    return;
  }
  replaceConstraintRowsKernel<<<static_cast<unsigned int>(rows.size()),
                                kThreads,
                                0,
                                static_cast<cudaStream_t>(ctx_.stream())>>>(
      rows.size(),
      rows.data(),
      mat_.rowPtrData(),
      mat_.colIndData(),
      mat_.valsData(),
      diag,
      nullptr,
      nullptr);
  cuda::checkLastError();
}

void CudaSystemMatrix::eliminateColumns(
    DeviceVectorView<const Index> rows,
    DeviceVectorView<const Real>  vals,
    DeviceVectorView<Real>        rhs)
{
  require(vals.size() == rows.size() && rhs.size() == mat_.rows(),
          "CUDA system matrix constraint vectors have incompatible dimensions");
  ensureConstraints(rows);
  if (rows.empty())
  {
    return;
  }

  eliminateConstraintColumnsKernel<<<static_cast<unsigned int>(mat_.rows()),
                                     kThreads,
                                     0,
                                     static_cast<cudaStream_t>(ctx_.stream())>>>(
      mat_.rows(),
      mat_.rowPtrData(),
      mat_.colIndData(),
      constraints_.row_to_constraint.data(),
      vals.data(),
      mat_.valsData(),
      rhs.data());
  cuda::checkLastError();
  replaceConstraintRowsKernel<<<static_cast<unsigned int>(rows.size()),
                                kThreads,
                                0,
                                static_cast<cudaStream_t>(ctx_.stream())>>>(
      rows.size(),
      rows.data(),
      mat_.rowPtrData(),
      mat_.colIndData(),
      mat_.valsData(),
      1.0,
      rhs.data(),
      vals.data());
  cuda::checkLastError();
}

} // namespace femx::linalg
