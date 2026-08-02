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
  for (Index k = 0; k < src.nnz(); ++k)
  {
    ++row_offsets[src.colIndData()[k] + 1];
  }
  for (Index i = 0; i < src.cols(); ++i)
  {
    row_offsets[i + 1] += row_offsets[i];
  }

  HostVector<Index> next = row_offsets;
  HostVector<Index> col_inds(src.nnz());
  HostVector<Real>  vals(src.nnz());
  for (Index i = 0; i < src.rows(); ++i)
  {
    for (Index k = src.rowPtrData()[i]; k < src.rowPtrData()[i + 1]; ++k)
    {
      const Index trans_row = src.colIndData()[k];
      const Index dst_entry = next[trans_row]++;
      col_inds[dst_entry]   = i;
      vals[dst_entry]       = src.valsData()[k];
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
  for (Index i = 0; i < mat.rows(); ++i)
  {
    Real val = 0.0;
    for (Index k = mat.rowPtrData()[i]; k < mat.rowPtrData()[i + 1]; ++k)
    {
      val += mat.valsData()[k] * dir[mat.colIndData()[k]];
    }
    out[i] = alpha * val + beta * out[i];
  }
}

void HostMatrixHandler::matvecT(const HostCsrMatrix&       mat,
                                HostVectorView<const Real> dir,
                                HostVectorView<Real>       out,
                                Real                       alpha,
                                Real                       beta) const
{
  checkCsrMatvec(mat, dir, out, true);
  for (Index j = 0; j < mat.cols(); ++j)
  {
    out[j] *= beta;
  }
  for (Index i = 0; i < mat.rows(); ++i)
  {
    const Real val = alpha * dir[i];
    for (Index k = mat.rowPtrData()[i]; k < mat.rowPtrData()[i + 1]; ++k)
    {
      out[mat.colIndData()[k]] += mat.valsData()[k] * val;
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
  for (Index i = 0; i < mat.rows(); ++i)
  {
    Real val = 0.0;
    for (Index j = 0; j < mat.cols(); ++j)
    {
      val += mat(i, j) * dir[j];
    }
    out[i] = alpha * val + beta * out[i];
  }
}

void HostMatrixHandler::matvecT(HostMatrixView<const Real> mat,
                                HostVectorView<const Real> dir,
                                HostVectorView<Real>       out,
                                Real                       alpha,
                                Real                       beta) const
{
  checkDenseMatvec(mat, dir, out, true);
  for (Index j = 0; j < mat.cols(); ++j)
  {
    Real val = 0.0;
    for (Index i = 0; i < mat.rows(); ++i)
    {
      val += mat(i, j) * dir[i];
    }
    out[j] = alpha * val + beta * out[j];
  }
}

} // namespace femx::linalg
