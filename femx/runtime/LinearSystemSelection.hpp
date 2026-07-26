#pragma once

namespace femx::runtime
{

/** @brief Identify the execution device independently of the solver. */
enum class ExecutionDevice
{
  Host,
  Device
};

/** @brief Identify a linear solver implementation independently of storage. */
enum class SolverType
{
  Dense,
  ReSolve,
  PETSc
};

/**
 * @brief Return the user-facing execution-device name.
 *
 * @param[in] device - Execution device.
 * @return Static lower-case device name.
 */
const char* name(ExecutionDevice device) noexcept;

/**
 * @brief Return the user-facing solver name.
 *
 * @param[in] solver - Linear solver implementation.
 * @return Static lower-case solver name.
 */
const char* name(SolverType solver) noexcept;

} // namespace femx::runtime
