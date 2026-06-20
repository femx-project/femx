#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "Components.hpp"
#include <femx/core/Types.hpp>

namespace femx::navier_var
{

struct MeshParams
{
  std::string file;
};

struct TimeParams
{
  Index steps = 40;
  Real  dt    = 0.025;
};

struct FluidConfig
{
  Real                rho = 1.0;
  std::optional<Real> mu;
  std::optional<Real> reynolds;
};

struct BoundarySelector
{
  Index       physical = 0;
  std::string name;
};

struct TargetParams
{
  std::string         type            = "poiseuille_pulse";
  std::string         quantity        = "mean_velocity";
  Real                bulk_speed      = 1.0;
  Real                pulse_amplitude = 0.35;
  Real                period          = 1.0;
  Real                radius          = 0.5;
  std::array<Real, 3> center          = {0.0, 0.5, 0.0};
  std::array<Real, 3> normal          = {1.0, 0.0, 0.0};
};

struct SolverParams
{
  std::string type    = "auto";
  std::string backend = "cuda";
};

struct OutputParams
{
  std::string basename = "ns-var";
};

struct BCsParams
{
  std::string                 name;
  Index                       physical = 0;
  std::string                 type     = "dirichlet";
  std::optional<Real>         ux;
  std::optional<Real>         uy;
  std::optional<Real>         uz;
  std::optional<Real>         p;
  std::optional<TargetParams> velocity;
};

struct ForwardParams
{
  MeshParams             mesh;
  TimeParams             time;
  FluidConfig            fluid;
  std::vector<BCsParams> bcs;
  OutputParams           output;
  SolverParams           solver;
};

struct RegularizationParams
{
  Real time = 1.0e-5;
  Real l2   = 1.0e-7;
};

struct OptimizerParams
{
  Index       max_iterations       = 40;
  Real        grad_abs_tolerance   = 5.0e-2;
  Real        grad_rel_tolerance   = 1.0e-8;
  Real        grad_step_tolerance  = 0.0;
  bool        use_options_database = true;
  std::string type                 = "lmvm";
};

struct BoundsParams
{
  Real                axial_min = 0.0;
  std::optional<Real> axial_max;
  Real                axial_max_scale = 1.2;
  bool                fix_non_axial   = true;
  std::array<Real, 3> normal          = {1.0, 0.0, 0.0};
};

struct InitialVelocityParams
{
  bool                enabled = false;
  std::optional<Real> lower;
  std::optional<Real> upper;
  Real                l2 = 0.0;
};

struct ObservationParams
{
  struct Grid
  {
    std::array<Real, 3>  lower       = {0.0, 0.0, 0.0};
    std::array<Real, 3>  upper       = {1.0, 1.0, 0.0};
    std::array<Real, 3>  origin      = {0.0, 0.0, 0.0};
    std::array<Real, 3>  spacing     = {1.0, 1.0, 1.0};
    std::array<Index, 3> counts      = {1, 1, 1};
    bool                 use_spacing = false;
  };

  std::string         type = "grid";
  std::string         file;
  std::optional<Grid> grid;
  std::vector<Index>  components;
};

struct InverseParams
{
  BoundarySelector      control{0, "inlet"};
  Real                  alpha = 1.0;
  RegularizationParams  reg;
  OptimizerParams       opt;
  BoundsParams          bounds;
  InitialVelocityParams initial_velocity;
  ObservationParams     obs;
};

struct Params
{
  ForwardParams forward;
  InverseParams inverse;
};

Params      loadConfig(const std::string& path);
FluidParams fluidParams(const Params& prm);
Real        Re(const Params& prm);

BoundarySelector              bcSelector(const BCsParams& bc);
const BCsParams&              controlBoundary(const Params& prm);
const TargetParams&           controlTarget(const Params& prm);
BoundarySelector              pressureGauge(const Params& prm);
std::vector<BoundarySelector> fixedVelocityBcs(const Params& prm);

} // namespace femx::navier_var
