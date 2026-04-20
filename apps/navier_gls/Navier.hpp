#pragma once

#include <vector>

#include <refem/common/Types.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{
class BlockFESpace;
class DirichletCondition;
class Mesh;
class SparseMatrix;
} // namespace refem

namespace refem
{

constexpr index_type dim = 2;

inline index_type nx       = 32;
inline index_type ny       = 32;
inline index_type steps    = 100;
inline index_type interval = 10;
inline real_type  dt       = 0.01;
inline real_type  max_cfl  = 10.0;
inline real_type  rho      = 1.0;
inline real_type  mu       = 0.01;
inline real_type  lid      = 1.0;

struct Snapshot
{
  real_type time{0.0};
  Vector    ux;
  Vector    uy;
  Vector    p;
};

DirichletCondition cavityBoundary(const BlockFESpace& space);

void assembleSystem(const BlockFESpace& space,
                    const Vector&       x,
                    const Vector&       xp,
                    bool                initial,
                    SparseMatrix&       A,
                    Vector&             b,
                    real_type&          max_cfl);

Snapshot makeSnapshot(const BlockFESpace& space,
                      const Vector&       x,
                      real_type           time);

} // namespace refem
