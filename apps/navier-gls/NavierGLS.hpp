#pragma once

#include <array>
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

inline index_type steps          = 100;
inline index_type interval       = 10;
inline real_type  dt             = 0.01;
inline real_type  rho            = 1.0;
inline real_type  mu             = 0.01;
inline real_type  lid            = 1.0;
inline real_type  inlet_velocity = 1.0;

struct BoundaryConditionSpec
{
  index_type  physical_tag = 0;
  std::string type         = "dirichlet";

  struct TimeProfile
  {
    std::string profile   = "constant";
    real_type   value     = 1.0;
    real_type   from      = 0.0;
    real_type   to        = 1.0;
    real_type   t0        = 0.0;
    real_type   t1        = 1.0;
    real_type   mean      = 1.0;
    real_type   amplitude = 0.0;
    real_type   frequency = 1.0;
    real_type   phase     = 0.0;
  };

  struct Value
  {
    std::string profile = "constant";
    real_type   value   = 0.0;
    index_type  axis    = 1;
    TimeProfile time;
  };

  struct FlowRate
  {
    std::vector<real_type>     time;
    std::vector<real_type>     value;
    real_type                  area        = 1.0;
    std::array<real_type, dim> normal      = {1.0, 0.0, 0.0};
    std::string                interpolate = "linear";
  };

  bool     has_ux       = false;
  bool     has_uy       = false;
  bool     has_uz       = false;
  bool     has_p        = false;
  bool     has_flowrate = false;
  Value    ux;
  Value    uy;
  Value    uz;
  Value    p;
  FlowRate flowrate;
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

DirichletCondition getBoundary(const BlockFESpace& space,
                               real_type           time = 0.0);

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
