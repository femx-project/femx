#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>

namespace femx
{
class CsrPattern;
class DenseMatrix;
class FESpace;
class MixedFESpace;

template <typename T>
class Vector;

namespace linalg
{
class MatrixBuilder;
} // namespace linalg

namespace assembly
{

enum class AssemblyMode
{
  Serial,
  Atomic
};

/**
 * @brief Build a sparse CSR pattern from a degree-of-freedom layout.
 *
 * The pattern contains the nonzero structure implied by element coupling, but
 * does not assign matrix values.
 */
CsrPattern makeCsrPattern(DofLayout layout);
CsrPattern makeCsrPattern(const FESpace& space);
CsrPattern makeCsrPattern(const MixedFESpace& space);

/**
 * @brief Scatter-add local finite-element vectors and matrices into globals.
 *
 * Assembler owns row/column dof layouts and provides the common FEM assembly
 * operations: initialize a global vector or matrix, then add one element-local
 * contribution at a time.
 */
class Assembler
{
public:
  explicit Assembler(DofLayout    space,
                     AssemblyMode mode = AssemblyMode::Serial);
  explicit Assembler(const FESpace& space,
                     AssemblyMode   mode = AssemblyMode::Serial);
  explicit Assembler(const MixedFESpace& space,
                     AssemblyMode        mode = AssemblyMode::Serial);

  Assembler(DofLayout    row_layout,
            DofLayout    col_layout,
            AssemblyMode mode = AssemblyMode::Serial);
  Assembler(const FESpace& row_space,
            const FESpace& col_space,
            AssemblyMode   mode = AssemblyMode::Serial);
  Assembler(const FESpace&      row_space,
            const MixedFESpace& col_space,
            AssemblyMode        mode = AssemblyMode::Serial);
  Assembler(const MixedFESpace& row_space,
            const FESpace&      col_space,
            AssemblyMode        mode = AssemblyMode::Serial);
  Assembler(const MixedFESpace& row_space,
            const MixedFESpace& col_space,
            AssemblyMode        mode = AssemblyMode::Serial);

  Index numElems() const;
  Index numRows() const;
  Index numCols() const;

  /** @brief Resize and zero a global vector compatible with the row layout. */
  void initVec(Vector<Real>& out) const;

  /** @brief Resize and zero a global matrix builder compatible with the layouts. */
  void initMat(linalg::MatrixBuilder& out) const;

  /** @brief Add one element-local vector to the global vector. */
  void addVec(Index ie, const Vector<Real>& local, Vector<Real>& out) const;

  /** @brief Add one element-local matrix to the global matrix builder. */
  void addMat(Index                  ie,
              const DenseMatrix&     local,
              linalg::MatrixBuilder& out) const;

private:
  void checkElemCounts() const;

  static void checkDof(Index id, Index size, const char* name);

private:
  DofLayout    row_layout_;
  DofLayout    col_layout_;
  bool         same_layout_{true};
  AssemblyMode mode_{AssemblyMode::Serial};
};

} // namespace assembly
} // namespace femx
