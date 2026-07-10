/**
 * @file Config.hpp
 * @author Kakeru Ueda (ueda.k.2290@m.isct.ac.jp)
 *
 */

#pragma once

#include <array>
#include <optional>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/Config.hpp>

#ifndef FEMX_NAVIERSTOKES_OUTPUT_DIR
#define FEMX_NAVIERSTOKES_OUTPUT_DIR "."
#endif

namespace femx::model::ns
{

struct ConvergenceParams
{
  bool  enabled     = false;  ///< Enable steady-state convergence check.
  Real  vel_rel_tol = 1.0e-8; ///< Relative velocity-change tolerance.
  Index min_steps   = 1;      ///< Minimum steps before convergence can stop.
};

struct TimeParams
{
  Index             steps = 100; ///< Number of time steps.
  Real              dt    = 0.01; ///< Time-step size.
  ConvergenceParams convergence;  ///< Optional convergence stopping criteria.
};

struct SolverParams
{
  std::string backend             = "cpu";       ///< Linear-solver backend.
  std::string method              = "iterative"; ///< Solver method family.
  std::string solve               = "fgmres";    ///< Krylov solve method.
  std::string preconditioner      = "ilu0";      ///< Preconditioner method.
  std::string gram_schmidt        = "cgs2";      ///< Orthogonalization method.
  std::string sketching           = "count";     ///< Sketching method.
  std::string preconditioner_side = "right";     ///< Preconditioner side.
  Index       max_iterations      = 5000;        ///< Maximum linear iterations.
  Index       restart             = 200;         ///< Krylov restart length.
  Real        relative_tolerance  = 1.0e-8;      ///< Linear residual tolerance.
  bool        flexible            = true;        ///< Enable flexible Krylov methods.
};

struct OutputParams
{
  Index       interval  = 10; ///< Field-output interval in time steps.
  std::string directory = FEMX_NAVIERSTOKES_OUTPUT_DIR; ///< Output directory.
};

struct VelocityProfileParams
{
  std::string                        type = "uniform"; ///< Profile type.
  Real                               rad  = 0.0;       ///< Profile radius.
  std::optional<std::array<Real, 3>> cen;              ///< Profile center.
};

struct VelocityParams
{
  Vector<Real>          time;              ///< Time samples.
  Vector<Real>          value;             ///< Velocity or flow-rate samples.
  Real                  area   = 1.0;      ///< Boundary area for flow-rate input.
  Real                  per    = 0.0;      ///< Period for pulse inputs.
  std::array<Real, 3>   nrm    = {1.0, 0.0, 0.0}; ///< Boundary normal.
  std::string           interp = "linear"; ///< Time interpolation method.
  std::string           qty    = "flowrate"; ///< Input quantity type.
  VelocityProfileParams prof;              ///< Spatial velocity profile.
};

struct BCsParams
{
  Index                         tag  = 0;           ///< Physical boundary tag.
  std::string                   type = "dirichlet"; ///< Boundary-condition type.
  std::optional<Real>           ux;                  ///< Prescribed x velocity.
  std::optional<Real>           uy;                  ///< Prescribed y velocity.
  std::optional<Real>           uz;                  ///< Prescribed z velocity.
  std::optional<Real>           p;                   ///< Prescribed pressure.
  std::optional<VelocityParams> velocity;            ///< Time-dependent velocity input.
};

struct Params
{
  std::string       mesh_file; ///< Mesh file path.
  TimeParams        time;      ///< Time-integration settings.
  FluidParams       fluid;     ///< Fluid material parameters.
  SolverParams      solver;    ///< Linear-solver settings.
  OutputParams      output;    ///< Output settings.
  Vector<BCsParams> bcs;       ///< Boundary-condition settings.
};

Params loadConfig(const std::string& path);

} // namespace femx::model::ns
