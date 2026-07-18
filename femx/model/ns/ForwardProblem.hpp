#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "ForwardConfig.hpp"
#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/TimeDirichletData.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/ForwardSolveMonitor.hpp>
#include <femx/model/ns/NavierStokesModel.hpp>

namespace femx
{
namespace state
{
class TimeIntegrator;
} // namespace state
} // namespace femx

namespace femx::model::ns
{

struct AppOptions
{
  std::string config_file;  ///< Input JSON config file.
  bool        help = false; ///< Print help and exit.
};

struct ForwardProblem
{
  explicit ForwardProblem(const Params& prm);

  ForwardProblem(const ForwardProblem&)            = delete;
  ForwardProblem& operator=(const ForwardProblem&) = delete;
  ForwardProblem(ForwardProblem&&)                 = delete;
  ForwardProblem& operator=(ForwardProblem&&)      = delete;

  NavierStokesModel                      model;
  fem::TimeDirichletData                 fixed;
  assembly::TimeDirichletControlResidual problem;
  HostVector                             x0;
  HostVector                             prm0;
};

AppOptions parseAppOptions(int   argc,
                           char* argv[],
                           bool  allow_unknown_options);

void printUsage(std::ostream&             out,
                const std::string&        executable,
                const std::string&        option_suffix = {},
                const Array<std::string>& extra_lines   = {});

std::unique_ptr<fem::FiniteElement> makeElem(const fem::Mesh&   mesh,
                                             const std::string& executable);

bool isFinite(const HostVector& x);

ForwardSolveResult solve(
    femx::state::TimeIntegrator& integrator,
    const ForwardProblem&        problem,
    const TimeParams&            time,
    const OutputParams&          prm,
    std::ostream*                terminal = nullptr,
    std::ostream*                log_out  = nullptr);

} // namespace femx::model::ns
