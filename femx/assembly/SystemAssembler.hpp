#pragma once

#include <femx/algebra/MatrixBuilder.hpp>
#include <femx/assembly/DofLayout.hpp>
#include <femx/core/Types.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/SystemVector.hpp>

namespace femx
{
namespace assembly
{

enum class AssemblyMode
{
  Serial,
  Atomic
};

/** @brief Scatter-adds elem vectors and matrices into equation objects. */
class SystemAssembler
{
public:
  explicit SystemAssembler(DofLayout    space,
                           AssemblyMode mode = AssemblyMode::Serial);

  explicit SystemAssembler(const FESpace& space,
                           AssemblyMode   mode = AssemblyMode::Serial);

  explicit SystemAssembler(const MixedFESpace& space,
                           AssemblyMode        mode = AssemblyMode::Serial);

  SystemAssembler(DofLayout    row_layout,
                  DofLayout    col_layout,
                  AssemblyMode mode = AssemblyMode::Serial);

  SystemAssembler(const FESpace& row_space,
                  const FESpace& col_space,
                  AssemblyMode   mode = AssemblyMode::Serial);

  SystemAssembler(const FESpace&      row_space,
                  const MixedFESpace& col_space,
                  AssemblyMode        mode = AssemblyMode::Serial);

  SystemAssembler(const MixedFESpace& row_space,
                  const FESpace&      col_space,
                  AssemblyMode        mode = AssemblyMode::Serial);

  SystemAssembler(const MixedFESpace& row_space,
                  const MixedFESpace& col_space,
                  AssemblyMode        mode = AssemblyMode::Serial);

  Index numCells() const;

  Index numRows() const;

  Index numCols() const;

  void initVec(Vector<Real>& out) const;

  void initVec(algebra::SystemVector& out) const;

  void initMat(algebra::MatrixBuilder& out) const;

  void addVec(Index ic, const Vector<Real>& local, Vector<Real>& out) const;

  void addVec(Index ic, const Vector<Real>& local, algebra::SystemVector& out) const;

  void addMat(Index ic, const DenseMatrix& local, algebra::MatrixBuilder& out) const;

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
