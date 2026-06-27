#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeStateMonitor.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace state
{

/** @brief Integrates a parameter-dependent state history in time. */
class TimeIntegrator
{
public:
  virtual ~TimeIntegrator() = default;

  virtual Index numSteps() const  = 0;
  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  void setMonitor(TimeStateMonitor* monitor)
  {
    monitor_ = monitor;
  }

  void clearMonitor()
  {
    monitor_ = nullptr;
  }

  virtual void solve(const Vector<Real>& prm,
                     TimeTrajectory&     tr) = 0;

protected:
  class MonitorScope
  {
  public:
    explicit MonitorScope(TimeIntegrator& integrator)
      : integrator_(&integrator)
    {
      integrator_->startMonitor();
    }

    MonitorScope(const MonitorScope&)            = delete;
    MonitorScope& operator=(const MonitorScope&) = delete;

    ~MonitorScope()
    {
      if (integrator_ != nullptr)
      {
        integrator_->stopMonitor();
      }
    }

  private:
    TimeIntegrator* integrator_{nullptr};
  };

  void observeState(Index               level,
                    const Vector<Real>& state)
  {
    if (monitor_ != nullptr)
    {
      monitor_->observe(level, state);
    }
  }

  bool observeStep(const TimeStepStateContext& ctx)
  {
    if (monitor_ != nullptr)
    {
      return monitor_->observeStep(ctx);
    }
    return false;
  }

private:
  void startMonitor()
  {
    if (monitor_ != nullptr)
    {
      monitor_->start(numSteps(), numStates());
    }
  }

  void stopMonitor()
  {
    if (monitor_ != nullptr)
    {
      monitor_->stop();
    }
  }

private:
  TimeStateMonitor* monitor_{nullptr};
};

} // namespace state
} // namespace femx
