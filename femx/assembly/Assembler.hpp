#pragma once

#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/MatrixBuilder.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>
#include <femx/fem/DofLayout.hpp>

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

  void initVector(Vector<Real>& out) const;

  void initMatrix(algebra::MatrixBuilder& out) const;

  void addVector(Index ic, const Vector<Real>& local, Vector<Real>& out) const;

  void addMatrix(Index ic,
                 const DenseMatrix& local,
                 algebra::MatrixBuilder& out) const;

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
