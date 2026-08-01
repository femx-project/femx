#include <algorithm>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/host/HostMatrixHandler.hpp>

namespace femx::linalg
{
namespace
{

void checkCsrMatvec(const HostCsrMatrix&       mat,
                    HostVectorView<const Real> dir,
                    HostVectorView<Real>       out,
                    bool                       trans)
{
  const Index in_size  = trans ? mat.rows() : mat.cols();
  const Index out_size = trans ? mat.cols() : mat.rows();
  require(dir.size() == in_size && out.size() == out_size,
          "Host CSR matvec vector size mismatch");
  require(!femx::detail::overlaps(dir, out),
          "Host CSR matvec does not support in-place views");
}

void checkDenseMatvec(HostMatrixView<const Real> mat,
                      HostVectorView<const Real> dir,
                      HostVectorView<Real>       out,
                      bool                       trans)
{
  const Index in_size  = trans ? mat.rows() : mat.cols();
  const Index out_size = trans ? mat.cols() : mat.rows();
  require(mat.rows() >= 0 && mat.cols() >= 0
              && dir.size() == in_size && out.size() == out_size
              && (mat.rows() * mat.cols() == 0 || mat.data() != nullptr),
          "Host dense matvec received incompatible storage");
  require(!femx::detail::overlaps(dir, out)
              && !femx::detail::overlaps(mat.data(),
                                         mat.rows() * mat.cols(),
                                         out.data(),
                                         out.size()),
          "Host dense matvec does not support aliased output");
}

} // namespace

void HostMatrixHandler::zero(HostCsrMatrix& mat) const
{
  std::fill(mat.vals().begin(), mat.vals().end(), 0.0);
}

void HostMatrixHandler::transpose(const HostCsrMatrix& src,
                                  HostCsrMatrix&       dst) const
{
  require(&src != &dst,
          "CSR transpose does not support in-place output");

  HostVector<Index> row_offsets(src.cols() + 1, 0);
  for (Index entry = 0; entry < src.nnz(); ++entry)
  {
    ++row_offsets[src.colIndData()[entry] + 1];
  }
  for (Index row = 0; row < src.cols(); ++row)
  {
    row_offsets[row + 1] += row_offsets[row];
  }

  HostVector<Index> next = row_offsets;
  HostVector<Index> col_inds(src.nnz());
  HostVector<Real>  vals(src.nnz());
  for (Index row = 0; row < src.rows(); ++row)
  {
    for (Index entry = src.rowPtrData()[row];
         entry < src.rowPtrData()[row + 1];
         ++entry)
    {
      const Index trans_row = src.colIndData()[entry];
      const Index dst_entry = next[trans_row]++;
      col_inds[dst_entry]   = row;
      vals[dst_entry]       = src.valsData()[entry];
    }
  }

  dst = HostCsrMatrix(HostCsrPattern(src.cols(),
                                     src.rows(),
                                     std::move(row_offsets),
                                     std::move(col_inds)));

  dst.vals() = vals;
}

void HostMatrixHandler::matvec(const HostCsrMatrix&       mat,
                               HostVectorView<const Real> dir,
                               HostVectorView<Real>       out,
                               Real                       alpha,
                               Real                       beta) const
{
  checkCsrMatvec(mat, dir, out, false);
  for (Index row = 0; row < mat.rows(); ++row)
  {
    Real val = 0.0;
    for (Index entry = mat.rowPtrData()[row];
         entry < mat.rowPtrData()[row + 1];
         ++entry)
    {
      val += mat.valsData()[entry] * dir[mat.colIndData()[entry]];
    }
    out[row] = alpha * val + beta * out[row];
  }
}

void HostMatrixHandler::matvecT(const HostCsrMatrix&       mat,
                                HostVectorView<const Real> dir,
                                HostVectorView<Real>       out,
                                Real                       alpha,
                                Real                       beta) const
{
  checkCsrMatvec(mat, dir, out, true);
  for (Index column = 0; column < mat.cols(); ++column)
  {
    out[column] *= beta;
  }
  for (Index row = 0; row < mat.rows(); ++row)
  {
    const Real val = alpha * dir[row];
    for (Index entry = mat.rowPtrData()[row];
         entry < mat.rowPtrData()[row + 1];
         ++entry)
    {
      out[mat.colIndData()[entry]] += mat.valsData()[entry] * val;
    }
  }
}

void HostMatrixHandler::matvec(HostMatrixView<const Real> mat,
                               HostVectorView<const Real> dir,
                               HostVectorView<Real>       out,
                               Real                       alpha,
                               Real                       beta) const
{
  checkDenseMatvec(mat, dir, out, false);
  for (Index row = 0; row < mat.rows(); ++row)
  {
    Real val = 0.0;
    for (Index col = 0; col < mat.cols(); ++col)
    {
      val += mat(row, col) * dir[col];
    }
    out[row] = alpha * val + beta * out[row];
  }
}

void HostMatrixHandler::matvecT(HostMatrixView<const Real> mat,
                                HostVectorView<const Real> dir,
                                HostVectorView<Real>       out,
                                Real                       alpha,
                                Real                       beta) const
{
  checkDenseMatvec(mat, dir, out, true);
  for (Index col = 0; col < mat.cols(); ++col)
  {
    Real val = 0.0;
    for (Index row = 0; row < mat.rows(); ++row)
    {
      val += mat(row, col) * dir[row];
    }
    out[col] = alpha * val + beta * out[col];
  }
}

} // namespace femx::linalg
