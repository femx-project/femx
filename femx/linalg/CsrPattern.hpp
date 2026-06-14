#pragma once

#include <vector>

#include <femx/common/Types.hpp>

namespace femx
{

class CsrPattern
{
public:
  CsrPattern(Index                                  rows,
             Index                                  cols,
             const std::vector<std::vector<Index>>& cdofs);

  ~CsrPattern();

  CsrPattern(const CsrPattern&)            = delete;
  CsrPattern& operator=(const CsrPattern&) = delete;

  CsrPattern(CsrPattern&& other) noexcept;
  CsrPattern& operator=(CsrPattern&& other) noexcept;

  Index rows() const;
  Index cols() const;
  Index nnz() const;

  Index numElems() const;
  Index numCooEntries() const;

  const Index* rowPtrData() const;
  const Index* colIndData() const;
  const Index* cooToCsrData() const;
  const Index* elemCooOffsetData() const;
  const Index* cellNumDofsData() const;

  Index mapToCsr(Index coo_index) const;
  Index elemCooOffset(Index ic) const;
  Index elemNumDofs(Index ic) const;

private:
  void countCooEntries(const std::vector<std::vector<Index>>& cdofs);
  void setupCooArrays(const std::vector<std::vector<Index>>& cdofs,
                      Index*                                 coo_rows,
                      Index*                                 coo_cols,
                      Index*                                 order);
  void setupCsrArrays(const Index* coo_rows,
                      const Index* coo_cols,
                      Index*       order);
  void release() noexcept;
  void moveFrom(CsrPattern&& other) noexcept;

private:
  Index num_rows_{0};
  Index num_cols_{0};
  Index nnz_{0};

  Index num_elems_{0};
  Index num_coo_entries_{0};

  Index* row_ptr_{nullptr};
  Index* col_ind_{nullptr};
  Index* map_to_csr_{nullptr};

  Index* elem_coo_offsets_{nullptr};
  Index* elem_num_dofs_{nullptr};
};

} // namespace femx
