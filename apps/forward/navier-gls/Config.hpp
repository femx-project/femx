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

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

#ifndef FEMX_NAVIERSTOKES_OUTPUT_DIR
#define FEMX_NAVIERSTOKES_OUTPUT_DIR "."
#endif

namespace femx
{

struct TimeParams
{
  Index steps = 100;
  Real  dt    = 0.01;
};

struct FluidParams
{
  Real rho = 1.0;
  Real mu  = 0.01;
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

struct FlowRateParams
{
  Vector              time;
  Vector              value;
  Real                area   = 1.0;
  std::array<Real, 3> normal = {1.0, 0.0, 0.0};
  std::string         interp = "linear";
};

struct BCsParams
{
  Index                         tag  = 0;
  std::string                   type = "dirichlet";
  std::optional<Real>           ux;
  std::optional<Real>           uy;
  std::optional<Real>           uz;
  std::optional<Real>           p;
  std::optional<FlowRateParams> flow;
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
