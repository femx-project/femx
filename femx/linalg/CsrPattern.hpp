#pragma once

#include <vector>

#include <femx/core/Types.hpp>

namespace femx
{

class CsrPattern
{
public:
  CsrPattern(index_type                                    rows,
             index_type                                    cols,
             const std::vector<std::vector<index_type>>& cdofs);

  ~CsrPattern();

  CsrPattern(const CsrPattern&)            = delete;
  CsrPattern& operator=(const CsrPattern&) = delete;

  CsrPattern(CsrPattern&& other) noexcept;
  CsrPattern& operator=(CsrPattern&& other) noexcept;

  index_type rows() const;
  index_type cols() const;
  index_type nnz() const;

  index_type numElems() const;
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
  void countCooEntries(const std::vector<std::vector<index_type>>& cdofs);
  void setupCooArrays(const std::vector<std::vector<index_type>>& cdofs,
                      index_type*                                 coo_rows,
                      index_type*                                 coo_cols,
                      index_type*                                 order);
  void setupCsrArrays(const index_type* coo_rows,
                      const index_type* coo_cols,
                      index_type*       order);
  void release() noexcept;
  void moveFrom(CsrPattern&& other) noexcept;

private:
  index_type num_rows_{0};
  index_type num_cols_{0};
  index_type nnz_{0};

  index_type num_elems_{0};
  index_type num_coo_entries_{0};

  index_type* row_ptr_{nullptr};
  index_type* col_ind_{nullptr};
  index_type* map_to_csr_{nullptr};

  index_type* elem_coo_offsets_{nullptr};
  index_type* elem_num_dofs_{nullptr};
};

} // namespace femx
