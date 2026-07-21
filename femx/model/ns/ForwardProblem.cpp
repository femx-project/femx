#include "ForwardProblem.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>

#include "Boundary.hpp"
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/TimeDirichletData.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>
#include <femx/model/ns/ForwardSolveMonitor.hpp>
#include <femx/model/ns/Helper.hpp>
#include <femx/runtime/Cli.hpp>
#include <femx/state/TimeIntegrator.hpp>
using namespace femx::state;
using namespace femx::assembly;

namespace femx::model::ns
{
namespace
{

fem::TimeDirichletData makeFixedDirichletData(
    const fem::MixedFESpace& space,
    const Array<BCsParams>&  bcs,
    Index                    steps,
    Real                     dt)
{
  return fem::makeTimeDirichletData(
      space.numDofs(),
      steps,
      dt,
      [&space, &bcs](Real time)
      {
        return makeDirichletBC(space, bcs, time);
      });
}

void configureMonitor(ForwardSolveMonitor& monitor,
                      const TimeParams&    time,
                      const OutputParams&  output,
                      std::ostream*        terminal,
                      std::ostream*        log_out)
{
  if (output.enabled)
  {
    monitor.setFieldOutput(output.directory, output.interval);
  }
  if (terminal != nullptr || log_out != nullptr)
  {
    monitor.setDetailedLog(terminal,
                           log_out,
                           time.convergence.enabled);
  }
  monitor.setConvergence({time.convergence.enabled,
                          time.convergence.vel_rel_tol,
                          time.convergence.min_steps});
}

} // namespace

ForwardProblem::ForwardProblem(const Params& prm)
  : model(prm.mesh_file, prm.time.steps, prm.time.dt, prm.fluid),
    fixed(makeFixedDirichletData(model.space(),
                                 prm.bcs,
                                 model.numSteps(),
                                 model.dt())),
    problem(model.residual(),
            fem::makeControlMap(model.numSteps(),
                                model.numStates(),
                                {},
                                fixed.dofs,
                                fixed.vals,
                                {},
                                0,
                                0)),
    x0(fixed.init_state),
    prm0(0)
{
}

AppOptions parseAppOptions(int   argc,
                           char* argv[],
                           bool  allow_unknown_options)
{
  AppOptions opts;

  for (int i = 1; i < argc; ++i)
  {
    const std::string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      opts.help = true;
      return opts;
    }
    if (key == "--config" || key == "-config")
    {
      opts.config_file = runtime::requireValue(argc, argv, i, key);
      continue;
    }
    if (!allow_unknown_options)
    {
      throw std::runtime_error("Unknown option: " + key);
    }
  }

  if (opts.config_file.empty())
  {
    throw std::runtime_error("Missing required option: --config FILE");
  }

  return opts;
}

void printUsage(std::ostream&             out,
                const std::string&        executable,
                const std::string&        option_suffix,
                const Array<std::string>& extra_lines)
{
  out << "Usage: " << executable << " --config FILE" << option_suffix << '\n';
  for (const std::string& line : extra_lines)
  {
    out << line << '\n';
  }
}

std::unique_ptr<fem::FiniteElement> makeElem(const fem::Mesh&   mesh,
                                             const std::string& executable)
{
  (void) executable;
  try
  {
    return makeElement(mesh);
  }
  catch (const std::runtime_error& e)
  {
    throw std::runtime_error(std::string(e.what()) + " (" + executable + ")");
  }
}

bool isFinite(const HostVector& x)
{
  for (Index i = 0; i < x.size(); ++i)
  {
    if (!std::isfinite(x[i]))
    {
      return false;
    }
  }
  return true;
}

template <class Backend>
ForwardSolveResult solveHost(TimeIntegrator<Backend>& integ,
                             const ForwardProblem&    prob,
                             const TimeParams&        time,
                             const OutputParams&      prm,
                             std::ostream*            terminal,
                             std::ostream*            log_out)
{
  ForwardSolveMonitor monitor(prob.model.space(),
                              prob.model.dt(),
                              prob.model.numSteps());
  configureMonitor(monitor, time, prm, terminal, log_out);

  monitor.start(integ.numSteps(), integ.numStates());
  typename TimeIntegrator<Backend>::Observer observer =
      [&monitor](const TimeStepStateContext& ctx)
  {
    if (ctx.level == 0)
    {
      monitor.observe(0, HostVector(ctx.curr));
      return false;
    }
    return monitor.observeStep(ctx);
  };

  try
  {
    integ.solve(prob.prm0.view(), observer);
  }
  catch (...)
  {
    monitor.stop();
    throw;
  }
  monitor.stop();
  return monitor.result();
}

ForwardSolveResult solve(HostTimeIntegrator&   integ,
                         const ForwardProblem& prob,
                         const TimeParams&     time,
                         const OutputParams&   prm,
                         std::ostream*         terminal,
                         std::ostream*         log_out)
{
  return solveHost(integ, prob, time, prm, terminal, log_out);
}

#if defined(FEMX_HAS_PETSC)
ForwardSolveResult solve(TimeIntegrator<linalg::PetscBackend>& integ,
                         const ForwardProblem&                 prob,
                         const TimeParams&                     time,
                         const OutputParams&                   prm,
                         std::ostream*                         terminal,
                         std::ostream*                         log_out)
{
  return solveHost(integ, prob, time, prm, terminal, log_out);
}
#endif

#if defined(FEMX_HAS_CUDA)
ForwardSolveResult solve(DeviceTimeIntegrator& integ,
                         const ForwardProblem& prob,
                         const TimeParams&     time,
                         const OutputParams&   prm,
                         std::ostream*         terminal,
                         std::ostream*         log_out)
{
  ForwardSolveMonitor monitor(prob.model.space(),
                              prob.model.dt(),
                              prob.model.numSteps());
  configureMonitor(monitor, time, prm, terminal, log_out);

  monitor.start(integ.numSteps(), integ.numStates());
  CudaContext               transfer;
  DeviceVector              parameters;
  linalg::CudaVectorHandler vec_handler(transfer);
  vec_handler.copy(prob.prm0, parameters);
  transfer.sync();

  DeviceTimeIntegrator::Observer observer =
      [&monitor](const TimeStepStateContext& ctx)
  {
    if (ctx.level == 0)
    {
      monitor.observe(0, HostVector(ctx.curr));
      return false;
    }
    return monitor.observeStep(ctx);
  };

  try
  {
    integ.solve(parameters.view(), std::move(observer));
  }
  catch (...)
  {
    monitor.stop();
    throw;
  }
  monitor.stop();
  return monitor.result();
}
#endif

} // namespace femx::model::ns
