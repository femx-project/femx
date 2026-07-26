#include <stdexcept>
#include <utility>

#include <femx/linalg/cuda/CudaLinearSystem.hpp>
#include <femx/linalg/native/HostLinearSystem.hpp>
#include <femx/runtime/LinearSystemFactory.hpp>

#if defined(FEMX_HAS_RESOLVE)
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#endif

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScLinearSystem.hpp>
#endif

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

bool supportsLinearSystem(ExecutionDevice device,
                          SolverType      solver) noexcept
{
  if (device == ExecutionDevice::Host && solver == SolverType::Dense)
  {
    return true;
  }
#if defined(FEMX_HAS_RESOLVE)
  if (solver == SolverType::ReSolve)
  {
    if (device == ExecutionDevice::Host)
    {
      return true;
    }
#if defined(FEMX_RESOLVE_USE_CUDA)
    return device == ExecutionDevice::Device;
#endif
  }
#endif
#if defined(FEMX_HAS_PETSC)
  if (device == ExecutionDevice::Host && solver == SolverType::PETSc)
  {
    return true;
  }
#endif
  return false;
}

std::unique_ptr<linalg::LinearSystem<MemorySpace::Host>>
makeHostLinearSystem(
    SolverType                                               solver,
    std::unique_ptr<linalg::LinearSolver<MemorySpace::Host>> native_solver)
{
  if (solver == SolverType::Dense)
  {
    if (native_solver)
    {
      return std::make_unique<linalg::HostLinearSystem>(
          std::move(native_solver));
    }
    return std::make_unique<linalg::HostLinearSystem>();
  }

  if (solver == SolverType::ReSolve)
  {
#if defined(FEMX_HAS_RESOLVE)
    if (!native_solver)
    {
      native_solver =
          std::make_unique<linalg::ReSolveLinearSolver>();
    }
    return std::make_unique<linalg::HostLinearSystem>(
        std::move(native_solver));
#else
    throw std::runtime_error(
        "ReSolve Host linear systems are unavailable in this build");
#endif
  }

  if (solver == SolverType::PETSc)
  {
    if (native_solver)
    {
      throw std::runtime_error(
          "PETSc linear systems own their native solver");
    }
#if defined(FEMX_HAS_PETSC)
    return std::make_unique<linalg::PETScLinearSystem>(
        PETSC_COMM_WORLD);
#else
    throw std::runtime_error(
        "PETSc linear systems are unavailable in this build");
#endif
  }

  throw std::runtime_error("Unknown Host linear-system solver selection");
}

std::unique_ptr<linalg::LinearSystem<MemorySpace::Device>>
makeDeviceLinearSystem(
    SolverType                                                 solver,
    std::unique_ptr<linalg::LinearSolver<MemorySpace::Device>> native_solver)
{
  if (solver != SolverType::ReSolve)
  {
    throw std::runtime_error(
        "Device execution requires the ReSolve solver");
  }

#if defined(FEMX_RESOLVE_USE_CUDA)
  if (!native_solver)
  {
    native_solver =
        std::make_unique<linalg::ReSolveLinearSolver>();
  }
  return std::make_unique<linalg::CudaLinearSystem>(
      std::move(native_solver));
#else
  static_cast<void>(native_solver);
  throw std::runtime_error(
      "ReSolve Device linear systems are unavailable in this build");
#endif
}

} // namespace femx::runtime
