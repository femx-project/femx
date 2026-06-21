#pragma once

#include <optional>
#include <string>
#include <vector>

#include "../ns-var/IO.hpp"

namespace femx::make_obs
{

struct OutputParams
{
  std::string file = "obs.txt";
  std::string vti_basename;
  std::string reference_basename;
  bool        write_vti       = false;
  bool        write_reference = true;
};

struct InputParams
{
  std::string trajectory;
  std::string velocity_field = "velocity";
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

struct ObservationCase
{
  std::string                       name;
  navier_var_new::ObservationParams obs;
  OutputParams                      output;
  NoiseParams                       noise;
  TimeSampleParams                  time;
};

struct Params
{
  InputParams                       input;
  navier_var_new::ObservationParams obs;
  OutputParams                      output;
  NoiseParams                       noise;
  TimeSampleParams                  time;
  std::vector<ObservationCase>      observations;
};

Params loadConfig(const std::string& path);

} // namespace femx::make_obs
