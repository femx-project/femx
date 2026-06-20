#pragma once

#include <femx/assembly/DofLayout.hpp>
#include <femx/assembly/ElementKernel.hpp>
#include <femx/core/Types.hpp>
#include <femx/problem/MatrixResidualEquation.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace assembly
{

/** @brief MatrixResidualEquation built from cell-local FEM kernels. */
class ElementResidualEquation final : public problem::MatrixResidualEquation
{
public:
  ElementResidualEquation(DofLayout            res_layout,
                          DofLayout            state_layout,
                          DofLayout            param_layout,
                          const ElementKernel& kernel);

  ElementResidualEquation(DofLayout            state_layout,
                          DofLayout            param_layout,
                          const ElementKernel& kernel);

  Index numStates() const override;

  Index numParams() const override;

  Index numRes() const override;

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override;

  void assembleStateJac(const Vector<Real>&   state,
                        const Vector<Real>&   prm,
                        algebra::SystemMatrix& out) const override;

  void assembleParamJac(const Vector<Real>&   state,
                        const Vector<Real>&   prm,
                        algebra::SystemMatrix& out) const override;

private:
  Index numCells() const;

  void checkCellCounts() const;

  void checkGlobalSizes(const Vector<Real>& state,
                        const Vector<Real>& prm) const;

  static void gather(const DofLayout&    layout,
                     const Vector<Real>& global,
                     Index               ic,
                     Vector<Real>&       local);

private:
  DofLayout            res_layout_;
  DofLayout            state_layout_;
  DofLayout            param_layout_;
  const ElementKernel& kernel_;
};

} // namespace assembly
} // namespace femx
