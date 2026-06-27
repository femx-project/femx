/**
 * @file Config.hpp
 * @author Kakeru Ueda (ueda.k.2290@m.isct.ac.jp)
 *
 */

#pragma once

#include <array>
#include <optional>
#include <string>

#include <femx/model/ns/Config.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

#ifndef FEMX_NAVIERSTOKES_OUTPUT_DIR
#define FEMX_NAVIERSTOKES_OUTPUT_DIR "."
#endif

namespace femx::model::ns
{


struct ConvergenceParams
{
  bool  enabled                     = false;
  Real  vel_rel_tol = 1.0e-8;
  Index min_steps                   = 1;
};

struct TimeParams
{
  Index             steps = 100;
  Real              dt    = 0.01;
  ConvergenceParams convergence;
};

struct SolverParams
{
  std::string backend             = "cpu";
  std::string method              = "iterative";
  std::string solve               = "fgmres";
  std::string preconditioner      = "ilu0";
  std::string gram_schmidt        = "cgs2";
  std::string sketching           = "count";
  std::string preconditioner_side = "right";
  Index       max_iterations      = 5000;
  Index       restart             = 200;
  Real        relative_tolerance  = 1.0e-8;
  bool        flexible            = true;
};

struct OutputParams
{
  Index       interval  = 10;
  std::string directory = FEMX_NAVIERSTOKES_OUTPUT_DIR;
};

struct VelocityProfileParams
{
  std::string                        type = "uniform";
  Real                               rad  = 0.0;
  std::optional<std::array<Real, 3>> cen;
};

struct VelocityParams
{
  Vector<Real>          time;
  Vector<Real>          value;
  Real                  area   = 1.0;
  Real                  per    = 0.0;
  std::array<Real, 3>   nrm    = {1.0, 0.0, 0.0};
  std::string           interp = "linear";
  std::string           qty    = "flowrate";
  VelocityProfileParams prof;
};

struct BCsParams
{
  Index                         tag  = 0;
  std::string                   type = "dirichlet";
  std::optional<Real>           ux;
  std::optional<Real>           uy;
  std::optional<Real>           uz;
  std::optional<Real>           p;
  std::optional<VelocityParams> velocity;
};

struct Params
{
  std::string       mesh_file;
  TimeParams        time;
  FluidParams       fluid;
  SolverParams      solver;
  OutputParams      output;
  Vector<BCsParams> bcs;
};

Params loadConfig(const std::string& path);

} // namespace femx::model::ns
