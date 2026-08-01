#pragma once

#include <iosfwd>

#include "Config.hpp"
#include "Monitor.hpp"
#include "NavierProblem.hpp"
#include <femx/state/TimeIntegrator.hpp>

namespace femx::apps::navier
{

/**
 * @brief Return whether every vector entry is finite.
 *
 * @param[in] vals - Values to inspect.
 * @return `true` when every entry is finite.
 */
bool hasFiniteValues(const HostVector<Real>& vals);

/**
 * @brief Run the Host time integrator and monitor accepted states.
 *
 * @param[in,out] integ    - Configured Host time integrator.
 * @param[in]     problem  - Navier-Stokes problem data.
 * @param[in]     time     - Time-stepping configuration.
 * @param[in]     out      - Field-output configuration.
 * @param[out]    terminal - Optional terminal log stream.
 * @param[out]    log_out  - Optional persistent log stream.
 * @return Final state and run summary.
 */
SolveResult solve(state::HostTimeIntegrator& integ,
                  const NavierProblem&       problem,
                  const TimeConfig&          time,
                  const OutputConfig&        out,
                  std::ostream*              terminal = nullptr,
                  std::ostream*              log_out  = nullptr);

#if defined(FEMX_HAS_CUDA)
/**
 * @brief Run the Device time integrator and monitor accepted states.
 *
 * @param[in,out] integ    - Configured Device time integrator.
 * @param[in]     problem  - Navier-Stokes problem data.
 * @param[in]     time     - Time-stepping configuration.
 * @param[in]     out      - Field-output configuration.
 * @param[out]    terminal - Optional terminal log stream.
 * @param[out]    log_out  - Optional persistent log stream.
 * @return Final state and run summary.
 */
SolveResult solve(state::DeviceTimeIntegrator& integ,
                  const NavierProblem&         problem,
                  const TimeConfig&            time,
                  const OutputConfig&          out,
                  std::ostream*                terminal = nullptr,
                  std::ostream*                log_out  = nullptr);
#endif

} // namespace femx::apps::navier
