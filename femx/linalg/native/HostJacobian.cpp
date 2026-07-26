#include <algorithm>
#include <cstdint>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/native/HostJacobian.hpp>

namespace femx::linalg
{
namespace
{

void checkElement(const ElementJacobianView& element)
{
  require(element.values.rows() == element.rows.size()
              && element.values.cols() == element.columns.size()
              && element.csr_entries.size()
                     == element.values.rows() * element.values.cols(),
          "Element Jacobian views have incompatible dimensions");
}

void checkCsrApply(const HostCsrMatrix&       matrix,
                   HostVectorView<const Real> direction,
                   HostVectorView<Real>       out,
                   bool                       transpose)
{
  const Index input_size  = transpose ? matrix.rows() : matrix.cols();
  const Index output_size = transpose ? matrix.cols() : matrix.rows();
  require(direction.size() == input_size && out.size() == output_size,
          "Host CSR application vector size mismatch");
  require(!femx::detail::overlaps(direction, out),
          "Host CSR application does not support in-place views");
}

void checkDenseApply(HostMatrixView<const Real> matrix,
                     HostVectorView<const Real> direction,
                     HostVectorView<Real>       out,
                     bool                       transpose)
{
  const Index input_size  = transpose ? matrix.rows() : matrix.cols();
  const Index output_size = transpose ? matrix.cols() : matrix.rows();
  require(matrix.rows() >= 0 && matrix.cols() >= 0
              && direction.size() == input_size && out.size() == output_size
              && (matrix.rows() * matrix.cols() == 0
                  || matrix.data() != nullptr),
          "Host dense application received incompatible storage");
  require(!femx::detail::overlaps(direction, out)
              && !femx::detail::overlaps(matrix.data(),
                                         matrix.rows() * matrix.cols(),
                                         out.data(),
                                         out.size()),
          "Host dense application does not support aliased output");
}

} // namespace

class HostJacobian::ConstraintCache
{
public:
  bool matches(const HostCsrPattern&       pattern,
               HostVectorView<const Index> rows) const
  {
    if (layout_id != pattern.layoutId()
        || constrained_rows.size() != rows.size())
    {
      return false;
    }
    for (Index i = 0; i < rows.size(); ++i)
    {
      if (constrained_rows[i] != rows[i])
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
            "Jacobian constraints require a square CSR pattern");

    layout_id = pattern.layoutId();
    constrained_rows.resize(rows.size());
    for (Index i = 0; i < rows.size(); ++i)
    {
      constrained_rows[i] = rows[i];
    }
    diagonal_entries.assign(rows.size(), -1);
    column_offsets.assign(rows.size() + 1, 0);
    row_to_constraint.assign(pattern.rows(), -1);

    for (Index ib = 0; ib < rows.size(); ++ib)
    {
      const Index row = rows[ib];
      require(row >= 0 && row < pattern.rows(),
              "Jacobian constrained row is out of range");
      require(row_to_constraint[row] < 0,
              "Jacobian constrained rows must be unique");
      row_to_constraint[row] = ib;
    }

    for (Index row = 0; row < pattern.rows(); ++row)
    {
      for (Index entry = pattern.rowPtrData()[row];
           entry < pattern.rowPtrData()[row + 1];
           ++entry)
      {
        const Index column = pattern.colIndData()[entry];
        const Index ib     = row_to_constraint[column];
        if (ib >= 0)
        {
          ++column_offsets[ib + 1];
          if (row == column)
          {
            require(diagonal_entries[ib] < 0,
                    "Jacobian constrained row has duplicate diagonal entries");
            diagonal_entries[ib] = entry;
          }
        }
      }
    }

    for (Index ib = 0; ib < rows.size(); ++ib)
    {
      require(diagonal_entries[ib] >= 0,
              "Jacobian constrained row has no diagonal entry");
      column_offsets[ib + 1] += column_offsets[ib];
    }

    column_entries.resize(column_offsets.back());
    column_rows.resize(column_offsets.back());
    HostVector<Index> next = column_offsets;
    for (Index row = 0; row < pattern.rows(); ++row)
    {
      for (Index entry = pattern.rowPtrData()[row];
           entry < pattern.rowPtrData()[row + 1];
           ++entry)
      {
        const Index ib = row_to_constraint[pattern.colIndData()[entry]];
        if (ib >= 0)
        {
          const Index destination     = next[ib]++;
          column_entries[destination] = entry;
          column_rows[destination]    = row;
        }
      }
    }
  }

  std::uint64_t     layout_id{0};
  HostVector<Index> constrained_rows;
  HostVector<Index> diagonal_entries;
  HostVector<Index> column_offsets;
  HostVector<Index> column_entries;
  HostVector<Index> column_rows;
  HostVector<Index> row_to_constraint;
};

HostJacobian::HostJacobian(Context<MemorySpace::Host>& ctx) noexcept
  : ctx_(ctx), constraints_(std::make_unique<ConstraintCache>())
{
}

HostJacobian::~HostJacobian() = default;

void HostJacobian::setup(const HostCsrPattern& pattern)
{
  if (matrix_.pattern().layoutId() != pattern.layoutId())
  {
    matrix_      = HostCsrMatrix(pattern);
    constraints_ = std::make_unique<ConstraintCache>();
  }
  else
  {
    ctx_.vectors().zero(matrix_.vals().view());
  }
}

void HostJacobian::addElement(const ElementJacobianView& element)
{
  checkElement(element);
  for (Index i = 0; i < element.csr_entries.size(); ++i)
  {
    const Index entry = element.csr_entries[i];
    require(entry >= 0 && entry < matrix_.nnz(),
            "Element Jacobian CSR entry is out of range");
#pragma omp atomic update
    matrix_.valsData()[entry] += element.values.data()[i];
  }
}

void HostJacobian::replaceRows(HostVectorView<const Index> rows,
                               Real                        diagonal)
{
  ConstraintCache& cache = constraints(rows);
  for (Index ib = 0; ib < rows.size(); ++ib)
  {
    const Index row = rows[ib];
    for (Index entry = matrix_.rowPtrData()[row];
         entry < matrix_.rowPtrData()[row + 1];
         ++entry)
    {
      matrix_.valsData()[entry] = 0.0;
    }
    matrix_.valsData()[cache.diagonal_entries[ib]] = diagonal;
  }
}

void HostJacobian::eliminateColumns(HostVectorView<const Index> rows,
                                    HostVectorView<const Real>  values,
                                    HostVectorView<Real>        rhs)
{
  require(values.size() == rows.size() && rhs.size() == matrix_.rows(),
          "Jacobian constraint vectors have incompatible dimensions");
  ConstraintCache& cache = constraints(rows);

  for (Index ib = 0; ib < rows.size(); ++ib)
  {
    for (Index i = cache.column_offsets[ib];
         i < cache.column_offsets[ib + 1];
         ++i)
    {
      const Index row   = cache.column_rows[i];
      const Index entry = cache.column_entries[i];
      if (cache.row_to_constraint[row] < 0)
      {
        rhs[row] -= matrix_.valsData()[entry] * values[ib];
      }
      matrix_.valsData()[entry] = 0.0;
    }
  }

  replaceRows(rows, 1.0);
  for (Index ib = 0; ib < rows.size(); ++ib)
  {
    rhs[rows[ib]] = values[ib];
  }
}

void HostJacobian::finalize()
{
}

void HostJacobian::apply(HostVectorView<const Real> direction,
                         HostVector<Real>&          out) const
{
  if (out.size() != matrix_.rows())
  {
    out.resize(matrix_.rows());
  }
  detail::applyHost(matrix_, direction, out.view());
}

void HostJacobian::applyT(HostVectorView<const Real> direction,
                          HostVector<Real>&          out) const
{
  if (out.size() != matrix_.cols())
  {
    out.resize(matrix_.cols());
  }
  detail::applyHostT(matrix_, direction, out.view());
}

const HostCsrMatrix& HostJacobian::matrix() const noexcept
{
  return matrix_;
}

void HostJacobian::transpose(const HostCsrMatrix& source,
                             HostCsrMatrix&       destination) const
{
  detail::transposeHostCsr(source, destination);
}

void HostJacobian::apply(const HostCsrMatrix&       matrix,
                         HostVectorView<const Real> direction,
                         HostVectorView<Real>       out,
                         Real                       alpha,
                         Real                       beta) const
{
  detail::applyHost(matrix, direction, out, alpha, beta);
}

void HostJacobian::applyT(const HostCsrMatrix&       matrix,
                          HostVectorView<const Real> direction,
                          HostVectorView<Real>       out,
                          Real                       alpha,
                          Real                       beta) const
{
  detail::applyHostT(matrix, direction, out, alpha, beta);
}

void HostJacobian::apply(HostMatrixView<const Real> matrix,
                         HostVectorView<const Real> direction,
                         HostVectorView<Real>       out,
                         Real                       alpha,
                         Real                       beta) const
{
  detail::applyHost(matrix, direction, out, alpha, beta);
}

void HostJacobian::applyT(HostMatrixView<const Real> matrix,
                          HostVectorView<const Real> direction,
                          HostVectorView<Real>       out,
                          Real                       alpha,
                          Real                       beta) const
{
  detail::applyHostT(matrix, direction, out, alpha, beta);
}

HostJacobian::ConstraintCache& HostJacobian::constraints(
    HostVectorView<const Index> rows)
{
  if (!constraints_->matches(matrix_.pattern(), rows))
  {
    constraints_->build(matrix_.pattern(), rows);
  }
  return *constraints_;
}

namespace detail
{

void applyHost(const HostCsrMatrix&       matrix,
               HostVectorView<const Real> direction,
               HostVectorView<Real>       out,
               Real                       alpha,
               Real                       beta)
{
  checkCsrApply(matrix, direction, out, false);
  for (Index row = 0; row < matrix.rows(); ++row)
  {
    Real value = 0.0;
    for (Index entry = matrix.rowPtrData()[row];
         entry < matrix.rowPtrData()[row + 1];
         ++entry)
    {
      value += matrix.valsData()[entry]
               * direction[matrix.colIndData()[entry]];
    }
    out[row] = alpha * value + beta * out[row];
  }
}

void applyHostT(const HostCsrMatrix&       matrix,
                HostVectorView<const Real> direction,
                HostVectorView<Real>       out,
                Real                       alpha,
                Real                       beta)
{
  checkCsrApply(matrix, direction, out, true);
  for (Index column = 0; column < matrix.cols(); ++column)
  {
    out[column] *= beta;
  }
  for (Index row = 0; row < matrix.rows(); ++row)
  {
    const Real value = alpha * direction[row];
    for (Index entry = matrix.rowPtrData()[row];
         entry < matrix.rowPtrData()[row + 1];
         ++entry)
    {
      out[matrix.colIndData()[entry]] += matrix.valsData()[entry] * value;
    }
  }
}

void applyHost(HostMatrixView<const Real> matrix,
               HostVectorView<const Real> direction,
               HostVectorView<Real>       out,
               Real                       alpha,
               Real                       beta)
{
  checkDenseApply(matrix, direction, out, false);
  for (Index row = 0; row < matrix.rows(); ++row)
  {
    Real value = 0.0;
    for (Index column = 0; column < matrix.cols(); ++column)
    {
      value += matrix(row, column) * direction[column];
    }
    out[row] = alpha * value + beta * out[row];
  }
}

void applyHostT(HostMatrixView<const Real> matrix,
                HostVectorView<const Real> direction,
                HostVectorView<Real>       out,
                Real                       alpha,
                Real                       beta)
{
  checkDenseApply(matrix, direction, out, true);
  for (Index column = 0; column < matrix.cols(); ++column)
  {
    Real value = 0.0;
    for (Index row = 0; row < matrix.rows(); ++row)
    {
      value += matrix(row, column) * direction[row];
    }
    out[column] = alpha * value + beta * out[column];
  }
}

void transposeHostCsr(const HostCsrMatrix& source,
                      HostCsrMatrix&       destination)
{
  require(&source != &destination,
          "CSR transpose does not support in-place output");

  HostVector<Index> row_offsets(source.cols() + 1, 0);
  for (Index entry = 0; entry < source.nnz(); ++entry)
  {
    ++row_offsets[source.colIndData()[entry] + 1];
  }
  for (Index row = 0; row < source.cols(); ++row)
  {
    row_offsets[row + 1] += row_offsets[row];
  }

  HostVector<Index> next = row_offsets;
  HostVector<Index> column_indices(source.nnz());
  HostVector<Real>  values(source.nnz());
  for (Index row = 0; row < source.rows(); ++row)
  {
    for (Index entry = source.rowPtrData()[row];
         entry < source.rowPtrData()[row + 1];
         ++entry)
    {
      const Index transpose_row         = source.colIndData()[entry];
      const Index destination_entry     = next[transpose_row]++;
      column_indices[destination_entry] = row;
      values[destination_entry]         = source.valsData()[entry];
    }
  }

  destination        = HostCsrMatrix(HostCsrPattern(source.cols(),
                                             source.rows(),
                                             std::move(row_offsets),
                                             std::move(column_indices)));
  destination.vals() = values;
}

} // namespace detail

} // namespace femx::linalg
