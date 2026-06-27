#pragma once

#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "Config.hpp"
#include <ForwardSolveMonitor.hpp>
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
namespace state
{
class TimeLinearIntegrator;
} // namespace state

struct AppOptions
{
  std::string          config_file;
  std::optional<Index> steps;
  bool                 help      = false;
  bool                 no_output = false;
};

struct BuildInfo
{
  Vector<std::pair<std::string, std::string>> entries;
};

struct FixedBoundaryValues
{
  Vector<Index> dofs;
  Vector<Real>  vals;
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
  assembly::TimeDirichletControlResidual problem;
  Vector<Real>                           x0;
  CsrPattern                             pettern;
  Vector<Real>                           prm0;
};

using ForwardSolveResult = navier::ForwardSolveResult;

AppOptions parseAppOptions(int   argc,
                           char* argv[],
                           bool  allow_unknown_options);

void printUsage(std::ostream&              out,
                const std::string&         executable,
                const std::string&         option_suffix = {},
                const Vector<std::string>& extra_lines   = {});

std::unique_ptr<FiniteElement> makeElem(const Mesh&        mesh,
                                        const std::string& executable);

bool isFinite(const Vector<Real>& x);

ForwardSolveResult solve(
    state::TimeLinearIntegrator& integrator,
    const ForwardProblem&         problem,
    const TimeParams&             time,
    const OutputParams&           prm,
    bool                          collect_output,
    std::ostream*                 terminal = nullptr,
    std::ostream*                 log_out  = nullptr);

void writeBuildInfo(const OutputParams& prm, const BuildInfo& info);

std::ofstream openRunLog(const OutputParams& prm);

} // namespace femx
