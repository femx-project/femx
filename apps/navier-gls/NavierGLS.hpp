#pragma once

#include <string>
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

constexpr index_type dim = 3;

inline index_type steps    = 100;
inline index_type interval = 10;
inline real_type  dt       = 0.01;
inline real_type  rho      = 1.0;
inline real_type  mu       = 0.01;
inline real_type  lid      = 1.0;
inline real_type  inlet_velocity = 1.0;

struct BoundaryConditionSpec
{
  index_type  physical_tag = 0;
  std::string type = "dirichlet";
  bool        has_ux = false;
  bool        has_uy = false;
  bool        has_uz = false;
  bool        has_p  = false;
  real_type   ux = 0.0;
  real_type   uy = 0.0;
  real_type   uz = 0.0;
  real_type   p  = 0.0;
};

inline std::vector<BoundaryConditionSpec> bcs;

struct Snapshot
{
  real_type time{0.0};
  Vector    ux;
  Vector    uy;
  Vector    uz;
  Vector    p;
};

DirichletCondition cavityBoundary(const BlockFESpace& space);

DirichletCondition navierBoundary(const BlockFESpace& space);

void assembleSystem(const BlockFESpace& space,
                    const Vector&       x,
                    const Vector&       xp,
                    bool                initial,
                    SparseMatrix&       A,
                    Vector&             b,
                    real_type&          observed_max_cfl);

Snapshot makeSnapshot(const BlockFESpace& space,
                      const Vector&       x,
                      real_type           time);

} // namespace refem
