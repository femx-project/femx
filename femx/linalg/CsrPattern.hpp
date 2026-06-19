#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/IndexSetList.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class CsrPattern
{
public:
  CsrPattern(Index               rows,
             Index               cols,
             const IndexSetList& cdofs);

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
  void countCooEntries(const IndexSetList& cdofs);
  void setupCooArrays(const IndexSetList& cdofs,
                      Vector<Index>&      coo_rows,
                      Vector<Index>&      coo_cols,
                      Vector<Index>&      order);
  void setupCsrArrays(const Vector<Index>& coo_rows,
                      const Vector<Index>& coo_cols,
                      Vector<Index>&       order);

private:
  Index num_rows_{0};
  Index num_cols_{0};
  Index nnz_{0};

  Index num_elems_{0};
  Index num_coo_entries_{0};

  std::vector<Index> row_ptr_;
  std::vector<Index> col_ind_;
  std::vector<Index> map_to_csr_;

  std::vector<Index> elem_coo_offsets_;
  std::vector<Index> elem_num_dofs_;
};

} // namespace femx
