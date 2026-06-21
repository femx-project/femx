/**
 * @file Config.hpp
 * @author Kakeru Ueda (ueda.k.2290@m.isct.ac.jp)
 *
 */

#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <NavierConfig.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

#ifndef FEMX_NAVIERSTOKES_OUTPUT_DIR
#define FEMX_NAVIERSTOKES_OUTPUT_DIR "."
#endif

namespace femx
{

using navier::FluidParams;

struct TimeParams
{
  Index steps = 100;
  Real  dt    = 0.01;
};

struct SolverParams
{
  std::string backend = "cpu";
};

struct OutputParams
{
  Index       interval  = 10;
  std::string directory = FEMX_NAVIERSTOKES_OUTPUT_DIR;
};

struct VelocityProfileParams
{
  std::string                        type   = "uniform";
  Real                               radius = 0.0;
  std::optional<std::array<Real, 3>> center;
};

struct VelocityParams
{
  Vector<Real>          time;
  Vector<Real>          value;
  Real                  area     = 1.0;
  Real                  period   = 0.0;
  std::array<Real, 3>   normal   = {1.0, 0.0, 0.0};
  std::string           interp   = "linear";
  std::string           quantity = "flowrate";
  VelocityProfileParams profile;
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
  std::string            mesh_file;
  TimeParams             time;
  FluidParams            fluid;
  SolverParams           solver;
  OutputParams           output;
  std::vector<BCsParams> bcs;
};

Params loadConfig(const std::string& path);

} // namespace femx
