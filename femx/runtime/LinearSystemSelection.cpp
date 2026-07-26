#include <femx/runtime/LinearSystemSelection.hpp>

namespace femx::runtime
{

const char* name(ExecutionDevice device) noexcept
{
  switch (device)
  {
  case ExecutionDevice::Host:
    return "host";
  case ExecutionDevice::Device:
    return "device";
  }
  return "unknown";
}

const char* name(SolverType solver) noexcept
{
  switch (solver)
  {
  case SolverType::Dense:
    return "dense";
  case SolverType::ReSolve:
    return "resolve";
  case SolverType::PETSc:
    return "petsc";
  }
  return "unknown";
}

} // namespace femx::runtime
