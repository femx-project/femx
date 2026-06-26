#pragma once

#include <array>
#include <optional>
#include <string>

#include <NavierConfig.hpp>
#include <femx/common/LinearInterpolation.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx::navier_en_var
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
  std::string         qty             = "mean_velocity";
  Real                bulk_speed      = 1.0;
  Real                pulse_amplitude = 0.35;
  Real                per             = 1.0;
  Real                rad             = 0.5;
  std::array<Real, 3> cen             = {0.0, 0.5, 0.0};
  std::array<Real, 3> nrm             = {1.0, 0.0, 0.0};
};

struct SolverParams
{
  std::string type    = "auto";
  std::string backend = "cuda";
  std::string method  = "iterative";
};

struct OutputParams
{
  std::string base = "ns-en-var";
};

struct VizOptions
{
  std::string base = "ns-en-var";
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
  MeshParams        mesh;
  TimeParams        time;
  FluidConfig       fluid;
  Vector<BCsParams> bcs;
  OutputParams      output;
  SolverParams      solver;
};

struct OptimizerParams
{
  Index       max_iterations = 40;
  Real        abs_tol        = 5.0e-2;
  Real        rel_tol        = 1.0e-8;
  Real        step_tol       = 0.0;
  std::string type           = "lmvm";
};

struct InitialVelocityParams
{
  bool enabled = false;
};

struct InitialGuessParams
{
  TimeParams        time;
  bool              has_time = false;
  Vector<BCsParams> bcs;
};

struct EnsembleParams
{
  std::string mean_file;
  std::string perturbations_file;
  std::string obs_mean_file;
  std::string obs_perturbations_file;
  Real        prior_weight = 1.0;
};

struct ObservationParams
{
  std::string file;
};

struct InverseParams
{
  BoundarySelector      ctr{0, "inlet"};
  Real                  alpha = 1.0;
  OptimizerParams       opt;
  InitialVelocityParams init_vel;
  InitialGuessParams    initial_guess;
  EnsembleParams        ens;
  ObservationParams     obs;
};

struct Params
{
  ForwardParams fwd;
  InverseParams inv;
};

Params      loadConfig(const std::string& path);
FluidParams fluidParams(const Params& prm);

BoundarySelector    bcSelector(const BCsParams& bc);
const BCsParams&    controlBoundary(const Params& prm);
const TargetParams& controlTarget(const Params& prm);

void writeResultViz(
    const Mesh&                        mesh,
    const MixedFESpace&                space,
    const DirichletControl&            ctr,
    const state::TimeTrajectory&       tr,
    const Vector<Real>&                prm,
    const Vector<LinearInterpolation>& ctr_time_stencils,
    Real                               dt,
    const VizOptions&                  opts,
    Real                               time_offset = 0.0);

} // namespace femx::navier_en_var
