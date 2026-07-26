#include "Solve.hpp"

#include <cmath>
#include <utility>

#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>

namespace femx::apps::ns_forward
{
namespace
{

void configureMonitor(Monitor&            monitor,
                      const TimeConfig&   time,
                      const OutputConfig& output,
                      std::ostream*       terminal,
                      std::ostream*       log_out)
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
  monitor.setConvergence(time.convergence);
}

SolveResult solveHost(state::HostTimeIntegrator& integrator,
                      const Problem&             problem,
                      const TimeConfig&          time,
                      const OutputConfig&        output,
                      std::ostream*              terminal,
                      std::ostream*              log_out)
{
  Monitor monitor(problem.model.space(),
                  problem.model.dt(),
                  problem.model.numSteps());
  configureMonitor(monitor, time, output, terminal, log_out);

  monitor.start(integrator.numSteps(), integrator.numStates());
  state::HostTimeIntegrator::Observer observer =
      [&monitor](const state::TimeStepStateContext& context)
  {
    if (context.level == 0)
    {
      monitor.observe(0, HostVector<Real>(context.curr));
      return false;
    }
    return monitor.observeStep(context);
  };

  try
  {
    integrator.solve(problem.parameters.view(), observer);
  }
  catch (...)
  {
    monitor.stop();
    throw;
  }
  monitor.stop();
  return monitor.result();
}

} // namespace

bool hasFiniteValues(const HostVector<Real>& values)
{
  for (Index i = 0; i < values.size(); ++i)
  {
    if (!std::isfinite(values[i]))
    {
      return false;
    }
  }
  return true;
}

SolveResult solve(state::HostTimeIntegrator& integrator,
                  const Problem&             problem,
                  const TimeConfig&          time,
                  const OutputConfig&        output,
                  std::ostream*              terminal,
                  std::ostream*              log_out)
{
  return solveHost(integrator, problem, time, output, terminal, log_out);
}

#if defined(FEMX_HAS_CUDA)
SolveResult solve(state::DeviceTimeIntegrator& integrator,
                  const Problem&               problem,
                  const TimeConfig&            time,
                  const OutputConfig&          output,
                  std::ostream*                terminal,
                  std::ostream*                log_out)
{
  Monitor monitor(problem.model.space(),
                  problem.model.dt(),
                  problem.model.numSteps());
  configureMonitor(monitor, time, output, terminal, log_out);

  monitor.start(integrator.numSteps(), integrator.numStates());
  linalg::CudaContext transfer;
  DeviceVector<Real>  parameters;
  auto&               vector_handler = transfer.vectors();
  vector_handler.copy(problem.parameters, parameters);
  transfer.sync();

  state::DeviceTimeIntegrator::Observer observer =
      [&monitor](const state::TimeStepStateContext& context)
  {
    if (context.level == 0)
    {
      monitor.observe(0, HostVector<Real>(context.curr));
      return false;
    }
    return monitor.observeStep(context);
  };

  try
  {
    integrator.solve(parameters.view(), std::move(observer));
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

} // namespace femx::apps::ns_forward
