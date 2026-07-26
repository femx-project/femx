#pragma once

#include <femx/common/Types.hpp>

namespace femx::runtime
{

/** @brief Identify a linear solver implementation independently of storage. */
enum class SolverType
{
  Dense,
  ReSolve,
  PETSc
};

/**
 * @brief Return the user-facing memory-space name.
 *
 * @param[in] space - Memory space.
 * @return Static lower-case memory-space name.
 */
const char* name(MemorySpace space) noexcept;

/**
 * @brief Return the user-facing solver name.
 *
 * @param[in] solver - Linear solver implementation.
 * @return Static lower-case solver name.
 */
const char* name(SolverType solver) noexcept;

} // namespace femx::runtime
