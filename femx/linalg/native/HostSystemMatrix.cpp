#include <algorithm>
#include <cstdint>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/native/HostSystemMatrix.hpp>

namespace femx::linalg
{
namespace
{

void checkElement(const ElementJacobianView& elem)
{
  require(elem.values.rows() == elem.rows.size()
              && elem.values.cols() == elem.columns.size()
              && elem.csr_entries.size()
                     == elem.values.rows() * elem.values.cols(),
          "Element Jacobian views have incompatible dimensions");
}

void checkCsrApply(const HostCsrMatrix&       mat,
                   HostVectorView<const Real> dir,
                   HostVectorView<Real>       out,
                   bool                       trans)
{
  const Index in_size  = trans ? mat.rows() : mat.cols();
  const Index out_size = trans ? mat.cols() : mat.rows();
  require(dir.size() == in_size && out.size() == out_size,
          "Host CSR application vector size mismatch");
  require(!femx::detail::overlaps(dir, out),
          "Host CSR application does not support in-place views");
}

void checkDenseApply(HostMatrixView<const Real> mat,
                     HostVectorView<const Real> dir,
                     HostVectorView<Real>       out,
                     bool                       trans)
{
  const Index in_size  = trans ? mat.rows() : mat.cols();
  const Index out_size = trans ? mat.cols() : mat.rows();
  require(mat.rows() >= 0 && mat.cols() >= 0
              && dir.size() == in_size && out.size() == out_size
              && (mat.rows() * mat.cols() == 0 || mat.data() != nullptr),
          "Host dense application received incompatible storage");
  require(!femx::detail::overlaps(dir, out)
              && !femx::detail::overlaps(mat.data(),
                                         mat.rows() * mat.cols(),
                                         out.data(),
                                         out.size()),
          "Host dense application does not support aliased output");
}

} // namespace

class HostSystemMatrix::ConstraintCache
{
public:
  bool matches(const HostCsrPattern&       pattern,
               HostVectorView<const Index> rows) const
  {
    if (layout_id != pattern.layoutId()
        || cached_rows.size() != rows.size())
    {
      return false;
    }
    for (Index i = 0; i < rows.size(); ++i)
    {
      if (cached_rows[i] != rows[i])
      {
        return false;
      }
    }
    return true;
  }

  void build(const HostCsrPattern&       pattern,
             HostVectorView<const Index> rows)
  {
    require(pattern.rows() == pattern.cols(),
            "System matrix constraints require a square CSR pattern");

    layout_id = pattern.layoutId();
    cached_rows.resize(rows.size());
    for (Index i = 0; i < rows.size(); ++i)
    {
      cached_rows[i] = rows[i];
    }
    diag_entries.assign(rows.size(), -1);
    col_offsets.assign(rows.size() + 1, 0);
    row_to_constraint.assign(pattern.rows(), -1);

    for (Index ib = 0; ib < rows.size(); ++ib)
    {
      const Index row = rows[ib];
      require(row >= 0 && row < pattern.rows(),
              "System matrix constrained row is out of range");
      require(row_to_constraint[row] < 0,
              "System matrix constrained rows must be unique");
      row_to_constraint[row] = ib;
    }

    for (Index row = 0; row < pattern.rows(); ++row)
    {
      for (Index entry = pattern.rowPtrData()[row];
           entry < pattern.rowPtrData()[row + 1];
           ++entry)
      {
        const Index col = pattern.colIndData()[entry];
        const Index ib  = row_to_constraint[col];
        if (ib >= 0)
        {
          ++col_offsets[ib + 1];
          if (row == col)
          {
            require(diag_entries[ib] < 0,
                    "System matrix constrained row has duplicate diagonal entries");
            diag_entries[ib] = entry;
          }
        }
      }
    }

    for (Index ib = 0; ib < rows.size(); ++ib)
    {
      require(diag_entries[ib] >= 0,
              "System matrix constrained row has no diagonal entry");
      col_offsets[ib + 1] += col_offsets[ib];
    }

    col_entries.resize(col_offsets.back());
    col_rows.resize(col_offsets.back());
    HostVector<Index> next = col_offsets;
    for (Index row = 0; row < pattern.rows(); ++row)
    {
      for (Index entry = pattern.rowPtrData()[row];
           entry < pattern.rowPtrData()[row + 1];
           ++entry)
      {
        const Index ib = row_to_constraint[pattern.colIndData()[entry]];
        if (ib >= 0)
        {
          const Index dst  = next[ib]++;
          col_entries[dst] = entry;
          col_rows[dst]    = row;
        }
      }
    }
  }

  std::uint64_t     layout_id{0};      ///< Cached CSR layout identifier.
  HostVector<Index> cached_rows;       ///< Cached constrained row indices.
  HostVector<Index> diag_entries;      ///< CSR diagonal entries by constraint.
  HostVector<Index> col_offsets;       ///< Constraint column-entry offsets.
  HostVector<Index> col_entries;       ///< CSR entries in constrained columns.
  HostVector<Index> col_rows;          ///< Rows of constrained-column entries.
  HostVector<Index> row_to_constraint; ///< Row-to-constraint mapping.
};

HostSystemMatrix::HostSystemMatrix(Context<MemorySpace::Host>& ctx) noexcept
  : ctx_(ctx), constraints_(std::make_unique<ConstraintCache>())
{
}

HostSystemMatrix::~HostSystemMatrix() = default;

void HostSystemMatrix::setup(const HostCsrPattern& pattern)
{
  if (mat_.pattern().layoutId() != pattern.layoutId())
  {
    mat_         = HostCsrMatrix(pattern);
    constraints_ = std::make_unique<ConstraintCache>();
  }
  else
  {
    ctx_.vectorHandler().zero(mat_.vals().view());
  }
}

void HostSystemMatrix::addElement(const ElementJacobianView& elem)
{
  checkElement(elem);
  for (Index i = 0; i < elem.csr_entries.size(); ++i)
  {
    const Index entry = elem.csr_entries[i];
    require(entry >= 0 && entry < mat_.nnz(),
            "Element Jacobian CSR entry is out of range");
#pragma omp atomic update
    mat_.valsData()[entry] += elem.values.data()[i];
  }
}

void HostSystemMatrix::replaceRows(HostVectorView<const Index> rows,
                                   Real                        diag)
{
  ConstraintCache& cache = constraints(rows);
  for (Index ib = 0; ib < rows.size(); ++ib)
  {
    const Index row = rows[ib];
    for (Index entry = mat_.rowPtrData()[row];
         entry < mat_.rowPtrData()[row + 1];
         ++entry)
    {
      mat_.valsData()[entry] = 0.0;
    }
    mat_.valsData()[cache.diag_entries[ib]] = diag;
  }
}

void HostSystemMatrix::eliminateColumns(HostVectorView<const Index> rows,
                                        HostVectorView<const Real>  vals,
                                        HostVectorView<Real>        rhs)
{
  require(vals.size() == rows.size() && rhs.size() == mat_.rows(),
          "System matrix constraint vectors have incompatible dimensions");
  ConstraintCache& cache = constraints(rows);

  for (Index ib = 0; ib < rows.size(); ++ib)
  {
    for (Index i = cache.col_offsets[ib];
         i < cache.col_offsets[ib + 1];
         ++i)
    {
      const Index row   = cache.col_rows[i];
      const Index entry = cache.col_entries[i];
      if (cache.row_to_constraint[row] < 0)
      {
        rhs[row] -= mat_.valsData()[entry] * vals[ib];
      }
      mat_.valsData()[entry] = 0.0;
    }
  }

  replaceRows(rows, 1.0);
  for (Index ib = 0; ib < rows.size(); ++ib)
  {
    rhs[rows[ib]] = vals[ib];
  }
}

void HostSystemMatrix::finalize()
{
}

void HostSystemMatrix::apply(HostVectorView<const Real> dir,
                             HostVector<Real>&          out) const
{
  if (out.size() != mat_.rows())
  {
    out.resize(mat_.rows());
  }
  detail::applyHost(mat_, dir, out.view());
}

void HostSystemMatrix::applyT(HostVectorView<const Real> dir,
                              HostVector<Real>&          out) const
{
  if (out.size() != mat_.cols())
  {
    out.resize(mat_.cols());
  }
  detail::applyHostT(mat_, dir, out.view());
}

const HostCsrMatrix& HostSystemMatrix::matrix() const noexcept
{
  return mat_;
}

void HostSystemMatrix::transpose(const HostCsrMatrix& src,
                                 HostCsrMatrix&       dst) const
{
  detail::transposeHostCsr(src, dst);
}

void HostSystemMatrix::apply(const HostCsrMatrix&       mat,
                             HostVectorView<const Real> dir,
                             HostVectorView<Real>       out,
                             Real                       alpha,
                             Real                       beta) const
{
  detail::applyHost(mat, dir, out, alpha, beta);
}

void HostSystemMatrix::applyT(const HostCsrMatrix&       mat,
                              HostVectorView<const Real> dir,
                              HostVectorView<Real>       out,
                              Real                       alpha,
                              Real                       beta) const
{
  detail::applyHostT(mat, dir, out, alpha, beta);
}

void HostSystemMatrix::apply(HostMatrixView<const Real> mat,
                             HostVectorView<const Real> dir,
                             HostVectorView<Real>       out,
                             Real                       alpha,
                             Real                       beta) const
{
  detail::applyHost(mat, dir, out, alpha, beta);
}

void HostSystemMatrix::applyT(HostMatrixView<const Real> mat,
                              HostVectorView<const Real> dir,
                              HostVectorView<Real>       out,
                              Real                       alpha,
                              Real                       beta) const
{
  detail::applyHostT(mat, dir, out, alpha, beta);
}

HostSystemMatrix::ConstraintCache& HostSystemMatrix::constraints(
    HostVectorView<const Index> rows)
{
  if (!constraints_->matches(mat_.pattern(), rows))
  {
    constraints_->build(mat_.pattern(), rows);
  }
  return *constraints_;
}

namespace detail
{

void applyHost(const HostCsrMatrix&       mat,
               HostVectorView<const Real> dir,
               HostVectorView<Real>       out,
               Real                       alpha,
               Real                       beta)
{
  checkCsrApply(mat, dir, out, false);
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

void applyHostT(const HostCsrMatrix&       mat,
                HostVectorView<const Real> dir,
                HostVectorView<Real>       out,
                Real                       alpha,
                Real                       beta)
{
  checkCsrApply(mat, dir, out, true);
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

void applyHost(HostMatrixView<const Real> mat,
               HostVectorView<const Real> dir,
               HostVectorView<Real>       out,
               Real                       alpha,
               Real                       beta)
{
  checkDenseApply(mat, dir, out, false);
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

void applyHostT(HostMatrixView<const Real> mat,
                HostVectorView<const Real> dir,
                HostVectorView<Real>       out,
                Real                       alpha,
                Real                       beta)
{
  checkDenseApply(mat, dir, out, true);
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

void transposeHostCsr(const HostCsrMatrix& src,
                      HostCsrMatrix&       dst)
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

  dst        = HostCsrMatrix(HostCsrPattern(src.cols(),
                                     src.rows(),
                                     std::move(row_offsets),
                                     std::move(col_inds)));
  dst.vals() = vals;
}

} // namespace detail

} // namespace femx::linalg
