#pragma once

#include <optional>
#include <string>

#include "../ns-var/Config.hpp"

namespace femx::make_obs
{

struct OutputParams
{
  std::string file = "obs.txt";
  std::string vti_basename;
  bool        write_vti = false;
};

struct NoiseParams
{
  bool                enabled = false;
  std::optional<Real> snr;
  std::optional<Real> snr_db;
};

struct TimeSampleParams
{
  std::optional<Real>  start_time;
  std::optional<Real>  end_time;
  std::optional<Index> start_level;
  std::optional<Index> end_level;
  std::optional<Index> num_points;
};

struct Params
{
  navier_var::ForwardParams     forward;
  navier_var::ObservationParams obs;
  OutputParams                  output;
  NoiseParams                   noise;
  TimeSampleParams              time;
};

Params loadConfig(const std::string& path);

} // namespace femx::make_obs
