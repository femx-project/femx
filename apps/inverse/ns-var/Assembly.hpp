#pragma once

#include "NavierStokesEquation.hpp"
#include <femx/core/Types.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/Vector.hpp>
#if defined(FEMX_HAS_PETSC)
#include <femx/algebra/backends/petsc/PETScSystemMatrix.hpp>
#endif

namespace femx
{
class ElementValues;
class GaussQuadrature;
class MixedFESpace;

namespace algebra
{
#if !defined(FEMX_HAS_PETSC)
class PETScSystemMatrix;
#endif
} // namespace algebra
} // namespace femx

namespace femx
{

struct NavierVarCellRange
{
  Index begin = 0;
  Index end   = 0;
};

void assembleElemResidual(const MixedFESpace&               space,
                          Index                             cell,
                          ElementValues&                    values,
                          const Vector<Real>&               x_next,
                          const Vector<Real>&               x,
                          const TimeNavierStokesParameters& prm,
                          Vector<Real>&                     out);

void assembleNextElemMatrix(const MixedFESpace&               space,
                            Index                             cell,
                            ElementValues&                    values,
                            const GaussQuadrature&            quad,
                            const Vector<Real>&               x_next,
                            const Vector<Real>&               x,
                            const TimeNavierStokesParameters& prm,
                            DenseMatrix&                      out);

void assemblePrevElemMatrix(const MixedFESpace&               space,
                            Index                             cell,
                            ElementValues&                    values,
                            const GaussQuadrature&            quad,
                            const Vector<Real>&               x_next,
                            const Vector<Real>&               x,
                            const TimeNavierStokesParameters& prm,
                            DenseMatrix&                      out);

#if defined(FEMX_HAS_PETSC)
void assembleNextStateJacPETSc(const MixedFESpace&               space,
                               const GaussQuadrature&            quad,
                               const Vector<Real>&               x_next,
                               const Vector<Real>&               x,
                               const TimeNavierStokesParameters& prm,
                               const NavierVarCellRange&         cells,
                               algebra::PETScSystemMatrix&       out);

void assemblePrevStateJacPETSc(const MixedFESpace&               space,
                               const GaussQuadrature&            quad,
                               const Vector<Real>&               x_next,
                               const Vector<Real>&               x,
                               const TimeNavierStokesParameters& prm,
                               const NavierVarCellRange&         cells,
                               algebra::PETScSystemMatrix&       out);
#endif

} // namespace femx
