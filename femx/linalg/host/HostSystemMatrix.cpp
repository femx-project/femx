#include <cstdint>

#include <femx/common/Checks.hpp>
#include <femx/linalg/host/HostSystemMatrix.hpp>

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

    for (Index i = 0; i < pattern.rows(); ++i)
    {
      for (Index k = pattern.rowPtrData()[i];
           k < pattern.rowPtrData()[i + 1];
           ++k)
      {
        const Index col = pattern.colIndData()[k];
        const Index ib  = row_to_constraint[col];
        if (ib >= 0)
        {
          ++col_offsets[ib + 1];
          if (i == col)
          {
            require(diag_entries[ib] < 0,
                    "System matrix constrained row has duplicate diagonal entries");
            diag_entries[ib] = k;
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
    for (Index i = 0; i < pattern.rows(); ++i)
    {
      for (Index k = pattern.rowPtrData()[i];
           k < pattern.rowPtrData()[i + 1];
           ++k)
      {
        const Index ib = row_to_constraint[pattern.colIndData()[k]];
        if (ib >= 0)
        {
          const Index dst  = next[ib]++;
          col_entries[dst] = k;
          col_rows[dst]    = i;
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
    ctx_.matrixHandler().zero(mat_);
  }
}

void HostSystemMatrix::addElement(const ElementJacobianView& elem)
{
  checkElement(elem);
  for (Index k = 0; k < elem.csr_entries.size(); ++k)
  {
    const Index entry = elem.csr_entries[k];
    require(entry >= 0 && entry < mat_.nnz(),
            "Element Jacobian CSR entry is out of range");
#pragma omp atomic update
    mat_.valsData()[entry] += elem.values.data()[k];
  }
}

void HostSystemMatrix::replaceRows(HostVectorView<const Index> rows,
                                   Real                        diag)
{
  ConstraintCache& cache = constraints(rows);
  for (Index ib = 0; ib < rows.size(); ++ib)
  {
    const Index row = rows[ib];
    for (Index k = mat_.rowPtrData()[row];
         k < mat_.rowPtrData()[row + 1];
         ++k)
    {
      mat_.valsData()[k] = 0.0;
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

void HostSystemMatrix::matvec(HostVectorView<const Real> dir,
                              HostVector<Real>&          out) const
{
  if (out.size() != mat_.rows())
  {
    out.resize(mat_.rows());
  }
  ctx_.matrixHandler().matvec(mat_, dir, out.view());
}

void HostSystemMatrix::matvecT(HostVectorView<const Real> dir,
                               HostVector<Real>&          out) const
{
  if (out.size() != mat_.cols())
  {
    out.resize(mat_.cols());
  }
  ctx_.matrixHandler().matvecT(mat_, dir, out.view());
}

const HostCsrMatrix& HostSystemMatrix::matrix() const noexcept
{
  return mat_;
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

} // namespace femx::linalg
