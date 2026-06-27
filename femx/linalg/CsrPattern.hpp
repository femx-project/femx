#pragma once

#include <functional>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class CsrPattern
{
public:
  using ElementDofProvider = std::function<void(Index, Vector<Index>&)>;

  CsrPattern(Index                     rows,
             Index                     cols,
             Index                     num_elems,
             const ElementDofProvider& elem_dofs);

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
  Index elemCooOffset(Index ie) const;
  Index elemNumDofs(Index ie) const;

private:
  void countCooEntries(const ElementDofProvider& elem_dofs);
  void setupCooArrays(const ElementDofProvider& elem_dofs,
                      Vector<Index>&            coo_rows,
                      Vector<Index>&            coo_cols,
                      Vector<Index>&            order);
  void setupCsrArrays(const Vector<Index>& coo_rows,
                      const Vector<Index>& coo_cols,
                      Vector<Index>&       order);

private:
  Index num_rows_{0};
  Index num_cols_{0};
  Index nnz_{0};

  Index ne_{0};
  Index num_coo_entries_{0};

  Vector<Index> row_ptr_;
  Vector<Index> col_ind_;
  Vector<Index> map_to_csr_;

  Vector<Index> elem_coo_offsets_;
  Vector<Index> elem_num_dofs_;
};

} // namespace femx
