#pragma once

#include <chrono>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Config.hpp"
#include <NavierKernel.hpp>
#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/assembly/TimeFEMResidual.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace solve
{
class TimeTrajectory;
}

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

struct FixedBoundaryValues
{
  Vector<Index> dofs;
  Vector<Real>  values;
};

struct ForwardProblem
{
  explicit ForwardProblem(const Params& prm);

  ForwardProblem(const ForwardProblem&)            = delete;
  ForwardProblem& operator=(const ForwardProblem&) = delete;
  ForwardProblem(ForwardProblem&&)                 = delete;
  ForwardProblem& operator=(ForwardProblem&&)      = delete;

  Index steps = 0;
  Real  dt    = 0.0;

  Mesh                                   mesh;
  std::unique_ptr<FiniteElement>         elem;
  MixedFESpace                           space;
  GaussQuadrature                        quad;
  navier::NavierKernel                   ns;
  assembly::TimeFEMResidual              fem;
  FixedBoundaryValues                    fixed;
  assembly::TimeDirichletControlResidual eq;
  Vector<Real>                           x0;
  CsrPattern                             pattern;
  Vector<Real>                           prm0;
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

void writeTrajectoryOutput(const ForwardProblem&        problem,
                           const solve::TimeTrajectory& tr,
                           const OutputParams&          prm);

void writeBuildInfo(const OutputParams& prm, const BuildInfo& info);

std::ofstream openRunLog(const OutputParams& prm);

} // namespace femx
