#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <NavierConfig.hpp>
#include <femx/common/LinearInterpolation.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/solve/TimeTrajectory.hpp>

namespace femx::navier_var_new
{

using navier::BoundarySelector;
using navier::FluidParams;

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
  std::optional<Real> Re;
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

struct VizOptions
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
  std::optional<TargetParams> vel;
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
  Real beta1 = 1.0e-5;
  Real beta2 = 1.0e-7;
  Real beta3 = 0.0;
  Real beta4 = 0.0;
};

struct OptimizerParams
{
  Index       max_iterations = 40;
  Real        abs_tol        = 5.0e-2;
  Real        rel_tol        = 1.0e-8;
  Real        step_tol       = 0.0;
  std::string type           = "lmvm";

  struct Scale
  {
    Real initial_velocity = 1.0;
    Real boundary         = 1.0;
  };

  Scale scale;
};

struct BoundsParams
{
  Real                min = 0.0;
  std::optional<Real> max;
  Real                max_scale = 1.2;
  std::array<Real, 3> normal    = {1.0, 0.0, 0.0};
};

struct InitialVelocityParams
{
  bool                enabled = false;
  std::optional<Real> lower;
  std::optional<Real> upper;
};

struct InitialGuessParams
{
  TimeParams             time;
  bool                   has_time = false;
  std::vector<BCsParams> bcs;
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
  BoundarySelector      ctr{0, "inlet"};
  Real                  alpha = 1.0;
  RegularizationParams  reg;
  OptimizerParams       opt;
  BoundsParams          bounds;
  InitialVelocityParams init_vel;
  InitialGuessParams    initial_guess;
  ObservationParams     obs;
};

struct Params
{
  ForwardParams fwd;
  InverseParams inv;
};

Params      loadConfig(const std::string& path);
FluidParams fluidParams(const Params& prm);

BoundarySelector              bcSelector(const BCsParams& bc);
const BCsParams&              controlBoundary(const Params& prm);
const TargetParams&           controlTarget(const Params& prm);

void writeResultViz(
    const Mesh&                       mesh,
    const MixedFESpace&               space,
    const DirichletControl&           control,
    const solve::TimeTrajectory&      tr,
    const Vector<Real>&               prm,
    const Vector<LinearInterpolation>& ctr_time_stencils,
    Real                              dt,
    const VizOptions&                 opts,
    Real                              time_offset = 0.0);

} // namespace femx::navier_var_new
