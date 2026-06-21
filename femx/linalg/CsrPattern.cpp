#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <femx/linalg/CsrPattern.hpp>

namespace femx
{

CsrPattern::CsrPattern(Index               rows,
                       Index               cols,
                       const IndexSetList& cdofs)
{
  if (rows < 0 || cols < 0)
  {
    throw std::runtime_error("CsrPattern dimensions must be non-negative");
  }

  num_rows_  = rows;
  num_cols_  = cols;
  num_elems_ = cdofs.numSets();

  elem_coo_offsets_.assign(static_cast<std::size_t>(num_elems_ + 1), 0);
  elem_num_dofs_.assign(static_cast<std::size_t>(num_elems_), 0);

  countCooEntries(cdofs);

  Vector<Index> coo_rows(num_coo_entries_);
  Vector<Index> coo_cols(num_coo_entries_);
  Vector<Index> order(num_coo_entries_);

  setupCooArrays(cdofs, coo_rows, coo_cols, order);
  setupCsrArrays(coo_rows, coo_cols, order);
}

void CsrPattern::countCooEntries(const IndexSetList& cdofs)
{
  num_coo_entries_ = 0;

  for (Index ic = 0; ic < num_elems_; ++ic)
  {
    const Index num_dofs = cdofs.setSize(ic);

    elem_num_dofs_[ic]  = num_dofs;
    num_coo_entries_   += num_dofs * num_dofs;
  }
}

void CsrPattern::setupCooArrays(
    const IndexSetList& cdofs,
    Vector<Index>&      coo_rows,
    Vector<Index>&      coo_cols,
    Vector<Index>&      order)
{
  Index counter = 0;

  for (Index ic = 0; ic < num_elems_; ++ic)
  {
    const Vector<Index> dofs     = cdofs.set(ic);
    const Index         num_dofs = elem_num_dofs_[ic];

    elem_coo_offsets_[ic] = counter;

    for (Index i = 0; i < num_dofs; ++i)
    {
      for (Index j = 0; j < num_dofs; ++j)
      {
        const Index row = dofs[i];
        const Index col = dofs[j];

        if (row < 0 || row >= num_rows_ || col < 0 || col >= num_cols_)
        {
          throw std::runtime_error("CsrPattern cell dof is out of range");
        }
        coo_rows[counter] = row;
        coo_cols[counter] = col;
        order[counter]    = counter;

        ++counter;
      }
    }
  }

  elem_coo_offsets_[num_elems_] = num_coo_entries_;
}

void CsrPattern::setupCsrArrays(const Vector<Index>& coo_rows,
                                const Vector<Index>& coo_cols,
                                Vector<Index>&       order)
{
  std::sort(order.begin(),
            order.end(),
            [&coo_rows, &coo_cols](Index a, Index b)
            {
              if (coo_rows[a] != coo_rows[b])
              {
                return coo_rows[a] < coo_rows[b];
              }
              return coo_cols[a] < coo_cols[b];
            });

  row_ptr_.assign(static_cast<std::size_t>(num_rows_ + 1), 0);
  map_to_csr_.assign(static_cast<std::size_t>(num_coo_entries_), 0);

  std::vector<Index> col_tmp;
  col_tmp.reserve(static_cast<std::size_t>(num_coo_entries_));
  nnz_ = 0;

  for (Index k = 0; k < num_coo_entries_; ++k)
  {
    const Index current = order[static_cast<std::size_t>(k)];
    const Index previous =
        k == 0 ? current : order[static_cast<std::size_t>(k - 1)];

    const bool is_new =
        k == 0 || coo_rows[current] != coo_rows[previous]
        || coo_cols[current] != coo_cols[previous];

    if (is_new)
    {
      col_tmp.push_back(coo_cols[current]);

      const Index row = coo_rows[current];
      ++row_ptr_[row + 1];

      ++nnz_;
    }

    map_to_csr_[current] = nnz_ - 1;
  }

  for (Index r = 0; r < num_rows_; ++r)
  {
    row_ptr_[r + 1] += row_ptr_[r];
  }

  col_ind_ = std::move(col_tmp);
}

Index CsrPattern::rows() const
{
  return num_rows_;
}

Index CsrPattern::cols() const
{
  return num_cols_;
}

Index CsrPattern::nnz() const
{
  return nnz_;
}

Index CsrPattern::numElems() const
{
  return num_elems_;
}

Index CsrPattern::numCooEntries() const
{
  return num_coo_entries_;
}

const Index* CsrPattern::rowPtrData() const
{
  return row_ptr_.data();
}

const Index* CsrPattern::colIndData() const
{
  return col_ind_.data();
}

const Index* CsrPattern::cooToCsrData() const
{
  return map_to_csr_.data();
}

const Index* CsrPattern::elemCooOffsetData() const
{
  return elem_coo_offsets_.data();
}

const Index* CsrPattern::cellNumDofsData() const
{
  return elem_num_dofs_.data();
}

Index CsrPattern::mapToCsr(Index i) const
{
  return map_to_csr_[static_cast<std::size_t>(i)];
}

Index CsrPattern::elemCooOffset(Index ic) const
{
  return elem_coo_offsets_[static_cast<std::size_t>(ic)];
}

Index CsrPattern::elemNumDofs(Index ic) const
{
  return elem_num_dofs_[static_cast<std::size_t>(ic)];
}

} // namespace femx
