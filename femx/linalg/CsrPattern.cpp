#include <algorithm>
#include <stdexcept>
#include <utility>

#include <femx/linalg/CsrPattern.hpp>

namespace femx
{

CsrPattern::CsrPattern(
    Index                                  rows,
    Index                                  cols,
    const std::vector<std::vector<Index>>& cdofs)
{
  if (rows < 0 || cols < 0)
  {
    throw std::runtime_error("CsrPattern dimensions must be non-negative");
  }

  num_rows_  = rows;
  num_cols_  = cols;
  num_elems_ = static_cast<Index>(cdofs.size());

  elem_coo_offsets_ = new Index[num_elems_ + 1];
  elem_num_dofs_    = new Index[num_elems_];

  countCooEntries(cdofs);

  Index* coo_rows = new Index[num_coo_entries_];
  Index* coo_cols = new Index[num_coo_entries_];
  Index* order    = new Index[num_coo_entries_];

  setupCooArrays(cdofs, coo_rows, coo_cols, order);
  setupCsrArrays(coo_rows, coo_cols, order);

  delete[] order;
  delete[] coo_cols;
  delete[] coo_rows;
}

CsrPattern::~CsrPattern()
{
  release();
}

CsrPattern::CsrPattern(CsrPattern&& other) noexcept
{
  moveFrom(std::move(other));
}

CsrPattern& CsrPattern::operator=(CsrPattern&& other) noexcept
{
  if (this != &other)
  {
    release();
    moveFrom(std::move(other));
  }
  return *this;
}

void CsrPattern::countCooEntries(
    const std::vector<std::vector<Index>>& cdofs)
{
  num_coo_entries_ = 0;

  for (Index ic = 0; ic < num_elems_; ++ic)
  {
    const auto& dofs  = cdofs[static_cast<std::size_t>(ic)];
    const auto  ndofs = static_cast<Index>(dofs.size());

    elem_num_dofs_[ic]  = ndofs;
    num_coo_entries_   += ndofs * ndofs;
  }
}

void CsrPattern::setupCooArrays(
    const std::vector<std::vector<Index>>& cdofs,
    Index*                                 coo_rows,
    Index*                                 coo_cols,
    Index*                                 order)
{
  Index counter = 0;

  for (Index ic = 0; ic < num_elems_; ++ic)
  {
    const auto& dofs  = cdofs[static_cast<std::size_t>(ic)];
    const Index ndofs = elem_num_dofs_[ic];

    elem_coo_offsets_[ic] = counter;

    for (Index i = 0; i < ndofs; ++i)
    {
      for (Index j = 0; j < ndofs; ++j)
      {
        const Index row = dofs[static_cast<std::size_t>(i)];
        const Index col = dofs[static_cast<std::size_t>(j)];
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

void CsrPattern::setupCsrArrays(const Index* coo_rows,
                                const Index* coo_cols,
                                Index*       order)
{
  std::sort(order,
            order + num_coo_entries_,
            [coo_rows, coo_cols](Index a, Index b)
            {
              if (coo_rows[a] != coo_rows[b])
              {
                return coo_rows[a] < coo_rows[b];
              }
              return coo_cols[a] < coo_cols[b];
            });

  row_ptr_    = new Index[num_rows_ + 1]();
  map_to_csr_ = new Index[num_coo_entries_];

  Index* col_tmp = new Index[num_coo_entries_];

  nnz_ = 0;

  for (Index k = 0; k < num_coo_entries_; ++k)
  {
    const Index current = order[k];

    const bool is_new =
        k == 0 || coo_rows[current] != coo_rows[order[k - 1]] || coo_cols[current] != coo_cols[order[k - 1]];

    if (is_new)
    {
      col_tmp[nnz_] = coo_cols[current];

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

  col_ind_ = new Index[nnz_];

  for (Index k = 0; k < nnz_; ++k)
  {
    col_ind_[k] = col_tmp[k];
  }

  delete[] col_tmp;
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
  return row_ptr_;
}

const Index* CsrPattern::colIndData() const
{
  return col_ind_;
}

const Index* CsrPattern::cooToCsrData() const
{
  return map_to_csr_;
}

const Index* CsrPattern::elemCooOffsetData() const
{
  return elem_coo_offsets_;
}

const Index* CsrPattern::cellNumDofsData() const
{
  return elem_num_dofs_;
}

Index CsrPattern::mapToCsr(Index i) const
{
  return map_to_csr_[i];
}

Index CsrPattern::elemCooOffset(Index ic) const
{
  return elem_coo_offsets_[ic];
}

Index CsrPattern::elemNumDofs(Index ic) const
{
  return elem_num_dofs_[ic];
}

void CsrPattern::release() noexcept
{
  delete[] row_ptr_;
  delete[] col_ind_;
  delete[] map_to_csr_;
  delete[] elem_coo_offsets_;
  delete[] elem_num_dofs_;

  row_ptr_          = nullptr;
  col_ind_          = nullptr;
  map_to_csr_       = nullptr;
  elem_coo_offsets_ = nullptr;
  elem_num_dofs_    = nullptr;
}

void CsrPattern::moveFrom(CsrPattern&& other) noexcept
{
  num_rows_         = other.num_rows_;
  num_cols_         = other.num_cols_;
  nnz_              = other.nnz_;
  num_elems_        = other.num_elems_;
  num_coo_entries_  = other.num_coo_entries_;
  row_ptr_          = other.row_ptr_;
  col_ind_          = other.col_ind_;
  map_to_csr_       = other.map_to_csr_;
  elem_coo_offsets_ = other.elem_coo_offsets_;
  elem_num_dofs_    = other.elem_num_dofs_;

  other.num_rows_         = 0;
  other.num_cols_         = 0;
  other.nnz_              = 0;
  other.num_elems_        = 0;
  other.num_coo_entries_  = 0;
  other.row_ptr_          = nullptr;
  other.col_ind_          = nullptr;
  other.map_to_csr_       = nullptr;
  other.elem_coo_offsets_ = nullptr;
  other.elem_num_dofs_    = nullptr;
}

} // namespace femx
