#pragma once

#include <array>
#include <vector>

#include "Components.hpp"
#include "Config.hpp"
#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
class ElementValues;
class MixedFESpace;

namespace system
{
class SparseSystemMatrix;
class PETScSystemMatrix;
class PETScSystemVector;
} // namespace system
} // namespace femx

namespace femx
{

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
                        const Vector&         x,
                        const Vector&         xp,
                        bool                  initial,
                        const FluidParams&    fluid,
                        Real                  dt,
                        DenseMatrix&          Ke,
                        Vector&               Fe,
                        Real&                 max_cfl);

void elemResidualFromSystem(const MixedFESpace& space,
                            Index               ic,
                            const DenseMatrix&  Ke,
                            const Vector&       Fe,
                            const Vector&       x_next,
                            Vector&             Re);

void assembleElemResidual(const MixedFESpace&   space,
                          Index                 ic,
                          ElementValues&        ev,
                          std::vector<QPState>& qps,
                          const Vector&         x_next,
                          const Vector&         x,
                          const Vector&         xp,
                          bool                  initial,
                          const FluidParams&    fluid,
                          Real                  dt,
                          Vector&               Re,
                          Real&                 max_cfl);

void assembleSystem(const MixedFESpace&         space,
                    const Vector&               x,
                    const Vector&               xp,
                    bool                        initial,
                    const FluidParams&          fluid,
                    Real                        dt,
                    system::SparseSystemMatrix& A,
                    Vector&                     b,
                    AssemblyStats&              stats);

void assembleSystem(const MixedFESpace&        space,
                    const Vector&              x,
                    const Vector&              xp,
                    bool                       initial,
                    const FluidParams&         fluid,
                    Real                       dt,
                    const CellRange&           cells,
                    system::PETScSystemMatrix& A,
                    system::PETScSystemVector& b,
                    AssemblyStats&             stats);

} // namespace femx
