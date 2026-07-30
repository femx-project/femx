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
#include <femx/common/Vector.hpp>
#include <femx/model/navier/FluidProperties.hpp>

#ifndef FEMX_NAVIER_OUTPUT_DIR
#define FEMX_NAVIER_OUTPUT_DIR "."
#endif

namespace femx::apps::navier
{

/** @brief Configure optional steady-state convergence detection. */
struct ConvergenceConfig
{
  bool  enabled     = false;  ///< Enable steady-state convergence check.
  Real  vel_rel_tol = 1.0e-8; ///< Relative velocity-change tolerance.
  Index min_steps   = 1;      ///< Minimum steps before convergence can stop.
};

/** @brief Configure the forward time interval. */
struct TimeConfig
{
  Index             steps = 100;  ///< Number of time steps.
  Real              dt    = 0.01; ///< Time-step size.
  ConvergenceConfig convergence;  ///< Optional convergence stopping criteria.
};

/** @brief Configure the selected linear solver. */
struct SolverConfig
{
  std::string method             = "iterative"; ///< Solver method family.
  std::string solve              = "fgmres";    ///< Krylov solve method.
  std::string preconditioner     = "ilu0";      ///< Preconditioner method.
  std::string gram_schmidt       = "cgs2";      ///< Orthogonalization method.
  std::string sketching          = "count";     ///< Sketching method.
  Index       max_itrs           = 5000;        ///< Maximum linear iterations.
  Index       restart            = 200;         ///< Krylov restart length.
  Real        relative_tolerance = 1.0e-8;      ///< Linear residual tolerance.
  bool        flexible           = true;        ///< Enable flexible Krylov methods.
};

/** @brief Configure field and log output. */
struct OutputConfig
{
  bool        enabled   = true;                   ///< Enable field and log output.
  Index       interval  = 10;                     ///< Field-output interval in time steps.
  std::string directory = FEMX_NAVIER_OUTPUT_DIR; ///< Output directory.
};

/** @brief Configure a spatial velocity profile. */
struct VelocityProfileConfig
{
  std::string                        type = "uniform"; ///< Profile type.
  Real                               rad  = 0.0;       ///< Profile radius.
  std::optional<std::array<Real, 3>> cen;              ///< Profile center.
};

/** @brief Configure a time-dependent boundary velocity. */
struct VelocityBoundaryConfig
{
  HostVector<Real>      time;                     ///< Time samples.
  HostVector<Real>      vals;                     ///< Velocity or flow-rate samples.
  Real                  area   = 1.0;             ///< Boundary area for flow-rate input.
  Real                  per    = 0.0;             ///< Period for pulse inputs.
  std::array<Real, 3>   nrm    = {1.0, 0.0, 0.0}; ///< Boundary normal.
  std::string           interp = "linear";        ///< Time interpolation method.
  std::string           qty    = "flowrate";      ///< Input quantity type.
  VelocityProfileConfig prof;                     ///< Spatial velocity profile.
};

/** @brief Configure one ordered boundary condition. */
struct BoundaryConditionConfig
{
  Index                                 tag  = 0;           ///< Physical boundary tag.
  std::string                           type = "dirichlet"; ///< Boundary-condition type.
  std::optional<Real>                   ux;                 ///< Prescribed x velocity.
  std::optional<Real>                   uy;                 ///< Prescribed y velocity.
  std::optional<Real>                   uz;                 ///< Prescribed z velocity.
  std::optional<Real>                   p;                  ///< Prescribed pressure.
  std::optional<VelocityBoundaryConfig> velocity;           ///< Time-dependent velocity input.
};

/** @brief Store one validated Navier-Stokes application configuration. */
struct Config
{
  std::string                         mesh_file;           ///< Mesh file path.
  TimeConfig                          time;                ///< Time-integration settings.
  model::navier::FluidProperties      fluid;               ///< Fluid material properties.
  SolverConfig                        solver;              ///< Linear-solver settings.
  OutputConfig                        output;              ///< Output settings.
  HostVector<BoundaryConditionConfig> boundary_conditions; ///< Boundary settings.
};

/**
 * @brief Load and validate a Navier-Stokes application configuration.
 *
 * @param[in] path - JSON configuration path.
 * @return Validated application configuration.
 * @throws std::runtime_error - If the file cannot be read or is invalid.
 */
Config loadConfig(const std::string& path);

} // namespace femx::apps::navier
