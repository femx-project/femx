#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
class FESpace;
class MixedFESpace;

namespace assembly
{

enum class AssemblyMode
{
  Serial,
  Atomic
};

/** @brief Scatter-adds local FEM vectors and matrices into global objects. */
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

  Index numCells() const;

  Index numRows() const;

  Index numCols() const;

  void initVec(Vector<Real>& out) const;

  void initMat(linalg::MatrixBuilder& out) const;

  void addVec(Index ic, const Vector<Real>& local, Vector<Real>& out) const;

  void addMat(Index                  ic,
              const DenseMatrix&     local,
              linalg::MatrixBuilder& out) const;

private:
  void checkCellCounts() const;

  static void checkDof(Index dof, Index size, const char* name);

private:
  DofLayout    row_layout_;
  DofLayout    col_layout_;
  bool         same_layout_{true};
  AssemblyMode mode_{AssemblyMode::Serial};
};

} // namespace assembly
} // namespace femx
