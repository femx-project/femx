#include <algorithm>

#include <refem/fe/BlockFESpace.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/linalg/FixedSparsityPattern.hpp>

namespace refem
{

FixedSparsityPattern::FixedSparsityPattern(const FESpace& space)
{
  num_rows_  = space.numDofs();
  num_cols_  = space.numDofs();
  num_elems_ = space.numElems();

  elem_coo_offsets_ = new index_type[num_elems_ + 1];
  elem_num_dofs_    = new index_type[num_elems_];

  countCooEntries(space);

  index_type* coo_rows = new index_type[num_coo_entries_];
  index_type* coo_cols = new index_type[num_coo_entries_];
  index_type* order    = new index_type[num_coo_entries_];

  setupCooArrays(space, coo_rows, coo_cols, order);
  setupCsrArrays(coo_rows, coo_cols, order);

  delete[] order;
  delete[] coo_cols;
  delete[] coo_rows;
}

FixedSparsityPattern::FixedSparsityPattern(const BlockFESpace& space)
{
  num_rows_  = space.numDofs();
  num_cols_  = space.numDofs();
  num_elems_ = space.numElems();

  elem_coo_offsets_ = new index_type[num_elems_ + 1];
  elem_num_dofs_    = new index_type[num_elems_];

  countCooEntries(space);

  index_type* coo_rows = new index_type[num_coo_entries_];
  index_type* coo_cols = new index_type[num_coo_entries_];
  index_type* order    = new index_type[num_coo_entries_];

  setupCooArrays(space, coo_rows, coo_cols, order);
  setupCsrArrays(coo_rows, coo_cols, order);

  delete[] order;
  delete[] coo_cols;
  delete[] coo_rows;
}

FixedSparsityPattern::FixedSparsityPattern(
    index_type                                  num_dofs,
    const std::vector<std::vector<index_type>>& cell_dofs)
{
  num_rows_  = num_dofs;
  num_cols_  = num_dofs;
  num_elems_ = static_cast<index_type>(cell_dofs.size());

  elem_coo_offsets_ = new index_type[num_elems_ + 1];
  elem_num_dofs_    = new index_type[num_elems_];

  countCooEntries(cell_dofs);

  index_type* coo_rows = new index_type[num_coo_entries_];
  index_type* coo_cols = new index_type[num_coo_entries_];
  index_type* order    = new index_type[num_coo_entries_];

  setupCooArrays(cell_dofs, coo_rows, coo_cols, order);
  setupCsrArrays(coo_rows, coo_cols, order);

  delete[] order;
  delete[] coo_cols;
  delete[] coo_rows;
}

FixedSparsityPattern::~FixedSparsityPattern()
{
  delete[] row_ptr_;
  delete[] col_ind_;
  delete[] map_to_csr_;
  delete[] elem_coo_offsets_;
  delete[] elem_num_dofs_;
}

void FixedSparsityPattern::countCooEntries(const FESpace& space)
{
  num_coo_entries_ = 0;

  for (index_type ic = 0; ic < num_elems_; ++ic)
  {
    const auto dofs  = space.elemDofs(ic);
    const auto ndofs = static_cast<index_type>(dofs.size());

    elem_num_dofs_[ic]  = ndofs;
    num_coo_entries_   += ndofs * ndofs;
  }
}

void FixedSparsityPattern::countCooEntries(const BlockFESpace& space)
{
  num_coo_entries_ = 0;

  for (index_type ic = 0; ic < num_elems_; ++ic)
  {
    const auto dofs  = space.elemDofs(ic);
    const auto ndofs = static_cast<index_type>(dofs.size());

    elem_num_dofs_[ic]  = ndofs;
    num_coo_entries_   += ndofs * ndofs;
  }
}

void FixedSparsityPattern::countCooEntries(
    const std::vector<std::vector<index_type>>& cell_dofs)
{
  num_coo_entries_ = 0;

  for (index_type ic = 0; ic < num_elems_; ++ic)
  {
    const auto ndofs =
        static_cast<index_type>(cell_dofs[static_cast<std::size_t>(ic)].size());

    elem_num_dofs_[ic]  = ndofs;
    num_coo_entries_   += ndofs * ndofs;
  }
}

void FixedSparsityPattern::setupCooArrays(const FESpace& space,
                                          index_type*    coo_rows,
                                          index_type*    coo_cols,
                                          index_type*    order)
{
  index_type counter = 0;

  for (index_type ic = 0; ic < num_elems_; ++ic)
  {
    const auto       dofs  = space.elemDofs(ic);
    const index_type ndofs = elem_num_dofs_[ic];

    elem_coo_offsets_[ic] = counter;

    for (index_type i = 0; i < ndofs; ++i)
    {
      for (index_type j = 0; j < ndofs; ++j)
      {
        coo_rows[counter] = dofs[i];
        coo_cols[counter] = dofs[j];
        order[counter]    = counter;

        ++counter;
      }
    }
  }

  elem_coo_offsets_[num_elems_] = num_coo_entries_;
}

void FixedSparsityPattern::setupCooArrays(const BlockFESpace& space,
                                          index_type*         coo_rows,
                                          index_type*         coo_cols,
                                          index_type*         order)
{
  index_type counter = 0;

  for (index_type ic = 0; ic < num_elems_; ++ic)
  {
    const auto       dofs  = space.elemDofs(ic);
    const index_type ndofs = elem_num_dofs_[ic];

    elem_coo_offsets_[ic] = counter;

    for (index_type i = 0; i < ndofs; ++i)
    {
      for (index_type j = 0; j < ndofs; ++j)
      {
        coo_rows[counter] = dofs[i];
        coo_cols[counter] = dofs[j];
        order[counter]    = counter;

        ++counter;
      }
    }
  }

  elem_coo_offsets_[num_elems_] = num_coo_entries_;
}

void FixedSparsityPattern::setupCooArrays(
    const std::vector<std::vector<index_type>>& cell_dofs,
    index_type*                                 coo_rows,
    index_type*                                 coo_cols,
    index_type*                                 order)
{
  index_type counter = 0;

  for (index_type ic = 0; ic < num_elems_; ++ic)
  {
    const auto&      dofs  = cell_dofs[static_cast<std::size_t>(ic)];
    const index_type ndofs = elem_num_dofs_[ic];

    elem_coo_offsets_[ic] = counter;

    for (index_type i = 0; i < ndofs; ++i)
    {
      for (index_type j = 0; j < ndofs; ++j)
      {
        coo_rows[counter] = dofs[static_cast<std::size_t>(i)];
        coo_cols[counter] = dofs[static_cast<std::size_t>(j)];
        order[counter]    = counter;

        ++counter;
      }
    }
  }

  elem_coo_offsets_[num_elems_] = num_coo_entries_;
}

void FixedSparsityPattern::setupCsrArrays(const index_type* coo_rows,
                                          const index_type* coo_cols,
                                          index_type*       order)
{
  std::sort(order,
            order + num_coo_entries_,
            [coo_rows, coo_cols](index_type a, index_type b)
            {
              if (coo_rows[a] != coo_rows[b])
              {
                return coo_rows[a] < coo_rows[b];
              }
              return coo_cols[a] < coo_cols[b];
            });

  row_ptr_    = new index_type[num_rows_ + 1]();
  map_to_csr_ = new index_type[num_coo_entries_];

  index_type* col_tmp = new index_type[num_coo_entries_];

  nnz_ = 0;

  for (index_type k = 0; k < num_coo_entries_; ++k)
  {
    const index_type current = order[k];

    const bool is_new =
        k == 0 || coo_rows[current] != coo_rows[order[k - 1]] || coo_cols[current] != coo_cols[order[k - 1]];

    if (is_new)
    {
      col_tmp[nnz_] = coo_cols[current];

      const index_type row = coo_rows[current];
      ++row_ptr_[row + 1];

      ++nnz_;
    }

    map_to_csr_[current] = nnz_ - 1;
  }

  for (index_type r = 0; r < num_rows_; ++r)
  {
    row_ptr_[r + 1] += row_ptr_[r];
  }

  col_ind_ = new index_type[nnz_];

  for (index_type k = 0; k < nnz_; ++k)
  {
    col_ind_[k] = col_tmp[k];
  }

  delete[] col_tmp;
}

index_type FixedSparsityPattern::rows() const
{
  return num_rows_;
}

index_type FixedSparsityPattern::cols() const
{
  return num_cols_;
}

index_type FixedSparsityPattern::nnz() const
{
  return nnz_;
}

index_type FixedSparsityPattern::numElems() const
{
  return num_elems_;
}

index_type FixedSparsityPattern::numCooEntries() const
{
  return num_coo_entries_;
}

const index_type* FixedSparsityPattern::rowPtrData() const
{
  return row_ptr_;
}

const index_type* FixedSparsityPattern::colIndData() const
{
  return col_ind_;
}

const index_type* FixedSparsityPattern::cooToCsrData() const
{
  return map_to_csr_;
}

const index_type* FixedSparsityPattern::elemCooOffsetData() const
{
  return elem_coo_offsets_;
}

const index_type* FixedSparsityPattern::cellNumDofsData() const
{
  return elem_num_dofs_;
}

index_type FixedSparsityPattern::mapToCsr(index_type i) const
{
  return map_to_csr_[i];
}

index_type FixedSparsityPattern::elemCooOffset(index_type i) const
{
  return elem_coo_offsets_[i];
}

index_type FixedSparsityPattern::elemNumDofs(index_type i) const
{
  return elem_num_dofs_[i];
}

} // namespace refem
