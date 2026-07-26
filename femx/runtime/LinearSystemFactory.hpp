#pragma once

#include <memory>

#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/LinearSystem.hpp>

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

/**
 * @brief Report whether the configured build supports a device/solver pair.
 *
 * @param[in] device - Execution device.
 * @param[in] solver - Linear solver implementation.
 * @return `true` when the pair can be constructed.
 */
bool supportsLinearSystem(ExecutionDevice device,
                          SolverType      solver) noexcept;

/**
 * @brief Construct a complete Host linear system.
 *
 * A supplied native solver is used by Dense or ReSolve systems. Passing no
 * solver selects the implementation's default configuration. PETSc owns its
 * native solver and therefore requires `native_solver` to be empty.
 *
 * @param[in] solver - Linear solver implementation.
 * @param[in] native_solver - Optional configured native Host solver.
 * @return Independently owned Host linear system.
 * @throws std::runtime_error - If the selection is unavailable or invalid.
 */
std::unique_ptr<linalg::LinearSystem<MemorySpace::Host>>
makeHostLinearSystem(
    SolverType                                               solver,
    std::unique_ptr<linalg::LinearSolver<MemorySpace::Host>> native_solver =
        {});

/**
 * @brief Construct a complete Device linear system.
 *
 * @param[in] solver - Linear solver implementation.
 * @param[in] native_solver - Optional configured native Device solver.
 * @return Independently owned Device linear system.
 * @throws std::runtime_error - If the selection is unavailable or invalid.
 */
std::unique_ptr<linalg::LinearSystem<MemorySpace::Device>>
makeDeviceLinearSystem(
    SolverType                                                 solver,
    std::unique_ptr<linalg::LinearSolver<MemorySpace::Device>> native_solver =
        {});

} // namespace femx::runtime
