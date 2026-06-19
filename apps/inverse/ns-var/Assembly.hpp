#pragma once

#include "NavierStokesEquation.hpp"
#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
class ElementValues;
class GaussQuadrature;
class MixedFESpace;

namespace system
{
class PETScSystemMatrix;
} // namespace system
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
                               system::PETScSystemMatrix&        out);

void assemblePrevStateJacPETSc(const MixedFESpace&               space,
                               const GaussQuadrature&            quad,
                               const Vector<Real>&               x_next,
                               const Vector<Real>&               x,
                               const TimeNavierStokesParameters& prm,
                               const NavierVarCellRange&         cells,
                               system::PETScSystemMatrix&        out);
#endif

} // namespace femx
