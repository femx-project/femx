#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>

namespace femx
{
class CsrPattern;
class DenseMatrix;

template <typename T>
class Vector;

namespace fem
{
class FESpace;
class MixedFESpace;
} // namespace fem

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
CsrPattern makeCsrPattern(fem::DofLayout layout);
CsrPattern makeCsrPattern(const fem::FESpace& space);
CsrPattern makeCsrPattern(const fem::MixedFESpace& space);

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
  explicit Assembler(fem::DofLayout space,
                     AssemblyMode   mode = AssemblyMode::Serial);
  explicit Assembler(const fem::FESpace& space,
                     AssemblyMode        mode = AssemblyMode::Serial);
  explicit Assembler(const fem::MixedFESpace& space,
                     AssemblyMode             mode = AssemblyMode::Serial);

  Assembler(fem::DofLayout row_layout,
            fem::DofLayout col_layout,
            AssemblyMode   mode = AssemblyMode::Serial);
  Assembler(const fem::FESpace& row_space,
            const fem::FESpace& col_space,
            AssemblyMode        mode = AssemblyMode::Serial);
  Assembler(const fem::FESpace&      row_space,
            const fem::MixedFESpace& col_space,
            AssemblyMode             mode = AssemblyMode::Serial);
  Assembler(const fem::MixedFESpace& row_space,
            const fem::FESpace&      col_space,
            AssemblyMode             mode = AssemblyMode::Serial);
  Assembler(const fem::MixedFESpace& row_space,
            const fem::MixedFESpace& col_space,
            AssemblyMode             mode = AssemblyMode::Serial);

  // Accessors
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
  fem::DofLayout row_layout_;                 ///< Element-to-global ids for assembled rows.
  fem::DofLayout col_layout_;                 ///< Element-to-global ids for assembled columns.
  bool           same_layout_{true};          ///< True when row and column layouts are identical.
  AssemblyMode   mode_{AssemblyMode::Serial}; ///< Scatter mode used during assembly.
};

} // namespace assembly
} // namespace femx
