#pragma once

#include <femx/common/Types.hpp>

namespace femx
{
template <typename T>
class Vector;

namespace state
{

struct TimeStepStateContext
{
  Index               level       = 0;
  Index               total_steps = 0;
  const Vector<Real>& previous;
  const Vector<Real>& current;
  Real                assembly_sec = 0.0;
  Real                solve_sec    = 0.0;
};

/**
 * @brief Non-owning sink for states produced by a time-marching solve.
 *
 * Implementations can record states or request early stopping after observing
 * timing and state context for each integration step.
 */
class TimeStateMonitor
{
public:
  virtual ~TimeStateMonitor() = default;

  virtual void start(Index num_steps,
                     Index num_states)
  {
    (void) num_steps;
    (void) num_states;
  }

  virtual void observe(Index               level,
                       const Vector<Real>& state) = 0;

  virtual bool observeStep(const TimeStepStateContext& ctx)
  {
    observe(ctx.level, ctx.current);
    return false;
  }

  virtual void stop()
  {
  }
};

} // namespace state
} // namespace femx
