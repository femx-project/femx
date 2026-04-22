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

#include <refem/common/Types.hpp>

#ifndef REFEM_NAVIERSTOKES_OUTPUT_DIR
#define REFEM_NAVIERSTOKES_OUTPUT_DIR "."
#endif

namespace refem
{

struct CLOptions
{
  std::string config_file;
};

struct TimeParams
{
  index_type steps = 100;
  real_type  dt    = 0.01;
};

struct FluidParams
{
  real_type rho = 1.0;
  real_type mu  = 0.01;
};

struct SolverParams
{
  std::string backend = "cpu";
};

struct OutputParams
{
  index_type  interval  = 10;
  std::string directory = REFEM_NAVIERSTOKES_OUTPUT_DIR;
};

struct FlowRateParams
{
  std::vector<real_type>   time;
  std::vector<real_type>   value;
  real_type                area        = 1.0;
  std::array<real_type, 3> normal      = {1.0, 0.0, 0.0};
  std::string              interpolate = "linear";
};

struct BCsParams
{
  index_type                    tag  = 0;
  std::string                   type = "dirichlet";
  std::optional<real_type>      ux;
  std::optional<real_type>      uy;
  std::optional<real_type>      uz;
  std::optional<real_type>      p;
  std::optional<FlowRateParams> flowrate;
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

CLOptions parseCommandLine(int argc, char* argv[]);

Params loadConfig(const std::string& path);

} // namespace refem
