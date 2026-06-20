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
#include <femx/algebra/Vector.hpp>

namespace femx
{

class FiniteElement;
class Mesh;
class MixedFESpace;

struct AppOptions
{
  std::string          config_file;
  std::optional<Index> steps;
  bool                 help      = false;
  bool                 no_output = false;
};

struct BuildInfo
{
  std::vector<std::pair<std::string, std::string>> entries;
};

struct Snapshot
{
  Real         time{0.0};
  Vector<Real> ux;
  Vector<Real> uy;
  Vector<Real> uz;
  Vector<Real> p;
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

bool isFinite(const Vector<Real>& x);

bool shouldWriteOutput(Index               step,
                       Index               num_steps,
                       const OutputParams& prm);

Snapshot makeSnapshot(const MixedFESpace& space,
                      const Vector<Real>& x,
                      Real                time);

void writeOutput(const Mesh&                  mesh,
                 const OutputParams&          prm,
                 const std::vector<Snapshot>& snapshots);

void writeBuildInfo(const OutputParams& prm, const BuildInfo& info);

std::ofstream openRunLog(const OutputParams& prm);

} // namespace femx
