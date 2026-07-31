#pragma once

namespace femx::runtime
{

/**
 * @brief Identify a linear solver implementation independently of storage.
 */
enum class SolverType
{
  Dense,
  ReSolve,
  PETSc
};

/**
 * @brief Return the user-facing solver name.
 *
 * @param[in] solver - Linear solver implementation.
 * @return Static lower-case solver name.
 */
const char* name(SolverType solver) noexcept;

} // namespace femx::runtime
