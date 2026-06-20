#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "Config.hpp"
#include "TrajectoryObservation.hpp"

#ifndef FEMX_MAKE_OBS_APP_NAME
#define FEMX_MAKE_OBS_APP_NAME "make-obs"
#endif

using namespace femx;
using namespace femx::make_obs;

namespace
{

struct AppOptions
{
  std::string         config_file;
  std::optional<femx::Real> snr;
  std::optional<femx::Real> snr_db;
  std::string         output_file;
  std::string         trajectory_file;
  bool                no_noise = false;
  bool                help     = false;
};

std::string requireValue(int                argc,
                         char**             argv,
                         int&               i,
                         const std::string& key)
{
  if (i + 1 >= argc)
  {
    throw std::runtime_error("Missing value for " + key);
  }
  return std::string(argv[++i]);
}

AppOptions parseAppOptions(int argc, char** argv)
{
  AppOptions options;
  for (int i = 1; i < argc; ++i)
  {
    const std::string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      options.help = true;
      return options;
    }
    if (key == "--config" || key == "-config")
    {
      options.config_file = requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--output")
    {
      options.output_file = requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--trajectory")
    {
      options.trajectory_file = requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--snr")
    {
      options.snr = std::stod(requireValue(argc, argv, i, key));
      continue;
    }
    if (key == "--snr-db")
    {
      options.snr_db = std::stod(requireValue(argc, argv, i, key));
      continue;
    }
    if (key == "--no-noise")
    {
      options.no_noise = true;
      continue;
    }
    throw std::runtime_error("Unknown option: " + key);
  }
  return options;
}

void printUsage(std::ostream& out)
{
  out << "Usage: " << FEMX_MAKE_OBS_APP_NAME
      << " --config FILE [--trajectory FILE] [--output FILE]"
      << " [--snr R | --snr-db DB] [--no-noise]\n";
}

void setNoise(femx::make_obs::NoiseParams& noise,
              const AppOptions&            options)
{
  if (options.no_noise)
  {
    noise.enabled = false;
    noise.snr.reset();
    noise.snr_db.reset();
  }
  if (options.snr)
  {
    noise.enabled = true;
    noise.snr     = *options.snr;
    noise.snr_db.reset();
  }
  if (options.snr_db)
  {
    noise.enabled = true;
    noise.snr_db  = *options.snr_db;
    noise.snr.reset();
  }
}

void applyOverrides(const AppOptions& options,
                    femx::make_obs::Params& prm)
{
  if (!options.trajectory_file.empty())
  {
    prm.input.trajectory = options.trajectory_file;
  }
  if (!options.output_file.empty())
  {
    if (prm.observations.empty())
    {
      prm.output.file = options.output_file;
    }
    else if (prm.observations.size() == 1)
    {
      prm.observations.front().output.file = options.output_file;
    }
    else
    {
      throw std::runtime_error(
          "--output is ambiguous when config has multiple observations");
    }
  }

  setNoise(prm.noise, options);
  for (auto& item : prm.observations)
  {
    setNoise(item.noise, options);
  }
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    const AppOptions options = parseAppOptions(argc, argv);
    if (options.help)
    {
      printUsage(std::cout);
      return 0;
    }
    if (options.config_file.empty())
    {
      throw std::runtime_error("--config FILE is required");
    }

    Params prm = loadConfig(options.config_file);
    applyOverrides(options, prm);
    writeTrajectoryObservationOutputs(prm);
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << FEMX_MAKE_OBS_APP_NAME << " failed: " << e.what()
              << '\n';
    return 1;
  }
}
