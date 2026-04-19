#pragma once

#include <refem/common/Types.hpp>

namespace refem
{

class FESpace;

class FixedSparsityPattern
{
public:
  explicit FixedSparsityPattern(const FESpace& space);

  ~FixedSparsityPattern();

  FixedSparsityPattern(const FixedSparsityPattern&)            = delete;
  FixedSparsityPattern& operator=(const FixedSparsityPattern&) = delete;

  index_type rows() const;
  index_type cols() const;
  index_type nnz() const;

  index_type numCells() const;
  index_type numCooEntries() const;

  const index_type* rowPtrData() const;
  const index_type* colIndData() const;
  const index_type* cooToCsrData() const;
  const index_type* elemCooOffsetData() const;
  const index_type* cellNumDofsData() const;

  index_type mapToCsr(index_type coo_index) const;
  index_type elemCooOffset(index_type cell) const;
  index_type elemNumDofs(index_type cell) const;

private:
  void countCooEntries(const FESpace& space);
  void setupCooArrays(const FESpace& space,
                      index_type*    coo_rows,
                      index_type*    coo_cols,
                      index_type*    order);
  void setupCsrArrays(const index_type* coo_rows,
                      const index_type* coo_cols,
                      index_type*       order);

private:
  index_type num_rows_{0};
  index_type num_cols_{0};
  index_type nnz_{0};

  index_type num_cells_{0};
  index_type num_coo_entries_{0};

  index_type* row_ptr_{nullptr};
  index_type* col_ind_{nullptr};
  index_type* map_to_csr_{nullptr};

  index_type* elem_coo_offsets_{nullptr};
  index_type* elem_num_dofs_{nullptr};
};

} // namespace refem
