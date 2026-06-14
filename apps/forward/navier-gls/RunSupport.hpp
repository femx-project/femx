#pragma once

#include <chrono>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Config.hpp"
#include <femx/core/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class FiniteElement;
class Mesh;
class MixedFESpace;

struct AppOptions
{
  std::string               config_file;
  std::optional<index_type> steps;
  bool                      help      = false;
  bool                      no_output = false;
};

struct BuildInfo
{
  std::vector<std::pair<std::string, std::string>> entries;
};

struct Snapshot
{
  real_type time{0.0};
  Vector    ux;
  Vector    uy;
  Vector    uz;
  Vector    p;
};

struct TimingStats
{
  double assembly = 0.0;
  double bc       = 0.0;
  double solve    = 0.0;
  double gather   = 0.0;
  double output   = 0.0;
  double total    = 0.0;
};

using Clock = std::chrono::high_resolution_clock;

double elapsedSeconds(Clock::time_point begin, Clock::time_point end);

template <typename Fn>
double timeBlock(Fn&& fn)
{
  const auto begin = Clock::now();
  std::forward<Fn>(fn)();
  return elapsedSeconds(begin, Clock::now());
}

AppOptions parseAppOptions(int   argc,
                           char* argv[],
                           bool  allow_unknown_options);

void printUsage(std::ostream&                   out,
                const std::string&              executable,
                const std::string&              option_suffix = {},
                const std::vector<std::string>& extra_lines   = {});

std::unique_ptr<FiniteElement> makeElem(const Mesh&        mesh,
                                        const std::string& executable);

bool isFinite(const Vector& x);

bool shouldWriteOutput(index_type          step,
                       index_type          num_steps,
                       const OutputParams& params);

Snapshot makeSnapshot(const MixedFESpace& space,
                      const Vector&       x,
                      real_type           time);

void writeOutput(const Mesh&                  mesh,
                 const OutputParams&          params,
                 const std::vector<Snapshot>& snapshots);

void writeBuildInfo(const OutputParams& params, const BuildInfo& info);

std::ofstream openRunLog(const OutputParams& params);

void writeTimingSummary(std::ostream&      out,
                        const TimingStats& timing,
                        index_type         steps,
                        index_type         ranks);

} // namespace femx
