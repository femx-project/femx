#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "Config.hpp"
#include "TrajectoryObservation.hpp"

using namespace std;
using namespace femx;

#ifndef FEMX_MAKE_OBS_APP_NAME
#define FEMX_MAKE_OBS_APP_NAME "make-obs"
#endif

using namespace make_obs;

namespace
{

struct AppOptions
{
  string         config_file;
  optional<Real> snr;
  optional<Real> snr_db;
  string         output_file;
  string         trajectory_file;
  bool           no_noise = false;
  bool           help     = false;
};

string requireValue(int           argc,
                    char**        argv,
                    int&          i,
                    const string& key)
{
  if (i + 1 >= argc)
  {
    throw runtime_error("Missing value for " + key);
  }
  return string(argv[++i]);
}

AppOptions parseAppOptions(int argc, char** argv)
{
  AppOptions opts;
  for (int i = 1; i < argc; ++i)
  {
    const string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      opts.help = true;
      return opts;
    }
    if (key == "--config" || key == "-config")
    {
      opts.config_file = requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--output")
    {
      opts.output_file = requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--trajectory")
    {
      opts.trajectory_file = requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--snr")
    {
      opts.snr = stod(requireValue(argc, argv, i, key));
      continue;
    }
    if (key == "--snr-db")
    {
      opts.snr_db = stod(requireValue(argc, argv, i, key));
      continue;
    }
    if (key == "--no-noise")
    {
      opts.no_noise = true;
      continue;
    }
    throw runtime_error("Unknown option: " + key);
  }
  return opts;
}

void printUsage(ostream& out)
{
  out << "Usage: " << FEMX_MAKE_OBS_APP_NAME
      << " --config FILE [--trajectory FILE] [--output FILE]"
      << " [--snr R | --snr-db DB] [--no-noise]\n";
}

void setNoise(make_obs::NoiseParams& noise,
              const AppOptions&      opts)
{
  if (opts.no_noise)
  {
    noise.enabled = false;
    noise.snr.reset();
    noise.snr_db.reset();
  }
  if (opts.snr)
  {
    noise.enabled = true;
    noise.snr     = *opts.snr;
    noise.snr_db.reset();
  }
  if (opts.snr_db)
  {
    noise.enabled = true;
    noise.snr_db  = *opts.snr_db;
    noise.snr.reset();
  }
}

void applyOverrides(const AppOptions& opts,
                    make_obs::Params& prm)
{
  if (!opts.trajectory_file.empty())
  {
    prm.input.tr = opts.trajectory_file;
  }
  if (!opts.output_file.empty())
  {
    if (prm.observations.empty())
    {
      prm.output.file = opts.output_file;
    }
    else if (prm.observations.size() == 1)
    {
      prm.observations.front().output.file = opts.output_file;
    }
    else
    {
      throw runtime_error(
          "--output is ambiguous when config has multiple observations");
    }
  }

  setNoise(prm.noise, opts);
  for (auto& item : prm.observations)
  {
    setNoise(item.noise, opts);
  }
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    const AppOptions opts = parseAppOptions(argc, argv);
    if (opts.help)
    {
      printUsage(cout);
      return 0;
    }
    if (opts.config_file.empty())
    {
      throw runtime_error("--config FILE is required");
    }

    Params prm = loadConfig(opts.config_file);
    applyOverrides(opts, prm);
    writeTrajectoryObservationOutputs(prm);
    return 0;
  }
  catch (const exception& e)
  {
    cerr << FEMX_MAKE_OBS_APP_NAME << " failed: " << e.what()
         << '\n';
    return 1;
  }
}
