#pragma once

#include <array>
#include <vector>

#include "Components.hpp"
#include "Config.hpp"
#include <femx/core/Types.hpp>
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
  real_type max_cfl = 0.0;
};

struct CellRange
{
  index_type begin = 0;
  index_type end   = 0;
};

void assembleElemSystem(const MixedFESpace&   space,
                        index_type            cell,
                        ElementValues&        ev,
                        std::vector<QPState>& qps,
                        const Vector&         x,
                        const Vector&         xp,
                        bool                  initial,
                        const FluidParams&    fluid,
                        real_type             dt,
                        DenseMatrix&          Ke,
                        Vector&               Fe,
                        real_type&            max_cfl);

void elemResidualFromSystem(const MixedFESpace& space,
                            index_type          cell,
                            const DenseMatrix&  Ke,
                            const Vector&       Fe,
                            const Vector&       x_next,
                            Vector&             Re);

void assembleElemResidual(const MixedFESpace&   space,
                          index_type            cell,
                          ElementValues&        ev,
                          std::vector<QPState>& qps,
                          const Vector&         x_next,
                          const Vector&         x,
                          const Vector&         xp,
                          bool                  initial,
                          const FluidParams&    fluid,
                          real_type             dt,
                          Vector&               Re,
                          real_type&            max_cfl);

void assembleSystem(const MixedFESpace&         space,
                    const Vector&               x,
                    const Vector&               xp,
                    bool                        initial,
                    const FluidParams&          fluid,
                    real_type                   dt,
                    system::SparseSystemMatrix& A,
                    Vector&                     b,
                    AssemblyStats&              stats);

void assembleSystem(const MixedFESpace&        space,
                    const Vector&              x,
                    const Vector&              xp,
                    bool                       initial,
                    const FluidParams&         fluid,
                    real_type                  dt,
                    const CellRange&           cells,
                    system::PETScSystemMatrix& A,
                    system::PETScSystemVector& b,
                    AssemblyStats&             stats);

} // namespace femx
