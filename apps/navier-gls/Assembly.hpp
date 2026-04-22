#pragma once

#include <array>

#include "Config.hpp"
#include <refem/common/Types.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{
class BlockFESpace;
class SparseMatrix;
} // namespace refem

namespace refem
{

struct AssemblyStats
{
  real_type max_cfl = 0.0;
};

void assembleSystem(const BlockFESpace& space,
                    const Vector&       x,
                    const Vector&       xp,
                    bool                initial,
                    const FluidParams&  fluid,
                    real_type           dt,
                    SparseMatrix&       A,
                    Vector&             b,
                    AssemblyStats&      stats);

} // namespace refem
