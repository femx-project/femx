#pragma once

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/MatrixOperator.hpp>

namespace femx
{
class DenseMatrix;

namespace linalg
{

/**
 * @brief Host CSR operator with element maps supplied by an AssemblyMap.
 *
 * The object owns its map, accumulates local matrices directly into the map's
 * immutable CSR graph, and applies the resulting numeric operator.
 */
class MapCsrMatrix final : public MatrixOperator
{
public:
  /** @brief Create an assembly target retaining the map's CSR graph. */
  explicit MapCsrMatrix(const assembly::HostAssemblyMap& map);

  Index numRows() const override;
  Index numCols() const override;

  void resize(Index rows, Index cols) override;
  void setZero() override;
  void set(Index row, Index col, Real val) override;
  void add(Index row, Index col, Real val) override;
  void addAtomic(Index row, Index col, Real val) override;
  void addElem(Index,
               const Array<Index>&,
               const Array<Index>&,
               const DenseMatrix&,
               bool) override;
  void finalize() override;

  void apply(const HostVector& dir, HostVector& out) const override;
  void applyT(const HostVector& dir, HostVector& out) const override;

  HostCsrMatrix&       mat();
  const HostCsrMatrix& mat() const;

private:
  void  checkElem(Index ie, const DenseMatrix& mat_e) const;
  Index findEntry(Index row, Index col) const;

  HostCsrMatrix             mat_;
  assembly::HostAssemblyMap map_;
};

} // namespace linalg
} // namespace femx
