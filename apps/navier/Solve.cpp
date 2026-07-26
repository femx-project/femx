#include "Solve.hpp"

#include <cmath>
#include <utility>

namespace femx::apps::navier
{
namespace
{

void configureMonitor(Monitor&            monitor,
                      const TimeConfig&   time,
                      const OutputConfig& out,
                      std::ostream*       terminal,
                      std::ostream*       log_out)
{
  if (out.enabled)
  {
    monitor.setFieldOutput(out.directory, out.interval);
  }
  if (terminal != nullptr || log_out != nullptr)
  {
    monitor.setDetailedLog(terminal,
                       log_out,
                       time.convergence.enabled);
  }
  monitor.setConvergence(time.convergence);
}

} // namespace

bool hasFiniteValues(const HostVector<Real>& vals)
{
  for (Index i = 0; i < vals.size(); ++i)
  {
    if (!std::isfinite(vals[i]))
    {
      return false;
    }
  }
  return true;
}

SolveResult solve(state::HostTimeIntegrator& integ,
                  const NavierStokesProblem& prob,
                  const TimeConfig&          time,
                  const OutputConfig&        out,
                  std::ostream*              terminal,
                  std::ostream*              log_out)
{
  Monitor monitor(prob.model().space(),
              prob.model().dt(),
              prob.model().numSteps());
  configureMonitor(monitor, time, out, terminal, log_out);

  monitor.start(integ.numSteps(), integ.numStates());
  state::HostTimeIntegrator::Observer observer =
      [&monitor](const state::TimeStepStateContext& ctx)
  {
    if (ctx.level == 0)
    {
      monitor.observe(0, HostVector<Real>(ctx.curr));
      return false;
    }
    return monitor.observeStep(ctx);
  };

  const HostVector<Real> prm;
  try
  {
    integ.solve(prm.view(), observer);
  }
  catch (...)
  {
    monitor.stop();
    throw;
  }
  monitor.stop();
  return monitor.result();
}

#if defined(FEMX_HAS_CUDA)
SolveResult solve(state::DeviceTimeIntegrator& integ,
                  const NavierStokesProblem&   prob,
                  const TimeConfig&            time,
                  const OutputConfig&          out,
                  std::ostream*                terminal,
                  std::ostream*                log_out)
{
  Monitor monitor(prob.model().space(),
              prob.model().dt(),
              prob.model().numSteps());
  configureMonitor(monitor, time, out, terminal, log_out);

  monitor.start(integ.numSteps(), integ.numStates());
  state::DeviceTimeIntegrator::Observer observer =
      [&monitor](const state::TimeStepStateContext& ctx)
  {
    if (ctx.level == 0)
    {
      monitor.observe(0, HostVector<Real>(ctx.curr));
      return false;
    }
    return monitor.observeStep(ctx);
  };

  const DeviceVector<Real> prm;
  try
  {
    integ.solve(prm.view(), std::move(observer));
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

} // namespace femx::apps::navier
