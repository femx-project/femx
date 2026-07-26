#pragma once

#include <iosfwd>

#include "Config.hpp"
#include "Monitor.hpp"
#include "Problem.hpp"
#include <femx/state/TimeIntegrator.hpp>

namespace femx::apps::ns_forward
{

/** @brief Return whether every vector entry is finite. */
bool hasFiniteValues(const HostVector<Real>& values);

SolveResult solve(state::HostTimeIntegrator& integrator,
                  const Problem&             problem,
                  const TimeConfig&          time,
                  const OutputConfig&        output,
                  std::ostream*              terminal = nullptr,
                  std::ostream*              log_out  = nullptr);

#if defined(FEMX_HAS_CUDA)
SolveResult solve(state::DeviceTimeIntegrator& integrator,
                  const Problem&               problem,
                  const TimeConfig&            time,
                  const OutputConfig&          output,
                  std::ostream*                terminal = nullptr,
                  std::ostream*                log_out  = nullptr);
#endif

} // namespace femx::apps::ns_forward
