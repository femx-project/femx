#pragma once

#include <vector>

#include <refem/common/Types.hpp>

namespace refem
{

class BlockFESpace;
class DenseMatrix;
class FESpace;
class FixedSparsityPattern;
class SparseMatrix;
class Vector;

class LocalAssembler
{
public:
  explicit LocalAssembler(const FESpace& space);
  explicit LocalAssembler(const BlockFESpace& space);
  LocalAssembler(const FESpace&               space,
                 const FixedSparsityPattern& pattern);
  LocalAssembler(const BlockFESpace&          space,
                 const FixedSparsityPattern& pattern);

  void addLocalMatrix(index_type         ic,
                      const DenseMatrix& Ke,
                      SparseMatrix&      A) const;

  void addLocalVector(index_type    ic,
                      const Vector& Fe,
                      Vector&       b);

private:
  void elemDofs(index_type ic);

private:
  const FESpace*              fe_space_{nullptr};
  const BlockFESpace*         block_space_{nullptr};
  const FixedSparsityPattern* pattern_{nullptr};
  std::vector<index_type>     elem_dofs_;
};

} // namespace refem
