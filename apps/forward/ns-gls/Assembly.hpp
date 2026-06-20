#pragma once

#include <array>
#include <vector>

#include "Components.hpp"
#include "Config.hpp"
#include <femx/core/Types.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/IndexSetList.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/backends/native/SparseSystemMatrix.hpp>
#if defined(FEMX_HAS_PETSC)
#include <femx/algebra/backends/petsc/PETScSystemMatrix.hpp>
#include <femx/algebra/backends/petsc/PETScSystemVector.hpp>
#endif

namespace femx
{
class ElementValues;
class MixedFESpace;

namespace algebra
{
#if !defined(FEMX_HAS_PETSC)
class PETScSystemMatrix;
class PETScSystemVector;
#endif
} // namespace algebra
} // namespace femx

namespace femx
{

using algebra::PETScSystemMatrix;
using algebra::PETScSystemVector;
using algebra::SparseSystemMatrix;

struct AssemblyStats
{
  Real max_cfl = 0.0;
};

struct CellRange
{
  Index begin = 0;
  Index end   = 0;
};

void assembleElemSystem(const MixedFESpace&   space,
                        Index                 ic,
                        ElementValues&        ev,
                        std::vector<QPState>& qps,
                        const Vector<Real>&   x,
                        const Vector<Real>&   xp,
                        bool                  initial,
                        const FluidParams&    fluid,
                        Real                  dt,
                        DenseMatrix&          Ke,
                        Vector<Real>&         Fe,
                        Real&                 max_cfl);

void elemResidualFromSystem(const MixedFESpace& space,
                            Index               ic,
                            const DenseMatrix&  Ke,
                            const Vector<Real>& Fe,
                            const Vector<Real>& x_next,
                            Vector<Real>&       Re);

void assembleElemResidual(const MixedFESpace&   space,
                          Index                 ic,
                          ElementValues&        ev,
                          std::vector<QPState>& qps,
                          const Vector<Real>&   x_next,
                          const Vector<Real>&   x,
                          const Vector<Real>&   xp,
                          bool                  initial,
                          const FluidParams&    fluid,
                          Real                  dt,
                          Vector<Real>&         Re,
                          Real&                 max_cfl);

void assembleSystem(const MixedFESpace& space,
                    const Vector<Real>& x,
                    const Vector<Real>& xp,
                    bool                initial,
                    const FluidParams&  fluid,
                    Real                dt,
                    SparseSystemMatrix& A,
                    Vector<Real>&       b,
                    AssemblyStats&      stats);

void assembleSystem(const MixedFESpace& space,
                    const Vector<Real>& x,
                    const Vector<Real>& xp,
                    bool                initial,
                    const FluidParams&  fluid,
                    Real                dt,
                    const IndexSetList& elem_dofs,
                    const CellRange&    cells,
                    PETScSystemMatrix&  A,
                    PETScSystemVector&  b,
                    AssemblyStats&      stats);

} // namespace femx
