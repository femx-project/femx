#pragma once

#include <memory>
#include <utility>

#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/linalg/LinearSystem.hpp>
#include <femx/runtime/LinearSystemFactory.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <pybind11/pybind11.h>

std::unique_ptr<femx::linalg::LinearSystem<femx::MemorySpace::Host>>
makePythonHostLinearSystem(femx::runtime::SolverType solver,
                           const pybind11::object&   options);

class PythonHostTimeIntegrator final
{
public:
  PythonHostTimeIntegrator(
      const femx::state::HostTimeResidual& res,
      femx::runtime::SolverType            solver,
      const pybind11::object&              options)
    : system_(makePythonHostLinearSystem(solver, options)),
      integ_(res, *system_)
  {
  }

  femx::state::HostTimeIntegrator& get() noexcept
  {
    return integ_;
  }

  const femx::state::HostTimeIntegrator& get() const noexcept
  {
    return integ_;
  }

private:
  std::unique_ptr<
      femx::linalg::LinearSystem<femx::MemorySpace::Host>>
                                  system_;
  femx::state::HostTimeIntegrator integ_;
};

class PythonTimeReducedFunctional
{
public:
  virtual ~PythonTimeReducedFunctional() = default;

  virtual femx::Index numParams() const noexcept = 0;

  virtual femx::Real value(
      femx::HostVectorView<const femx::Real> prm,
      femx::inverse::TimeReducedProgress     progress = {}) = 0;
  virtual void grad(
      femx::HostVectorView<const femx::Real> prm,
      femx::HostVectorView<femx::Real>       out,
      femx::inverse::TimeReducedProgress     progress = {}) = 0;
  virtual femx::Real valueGrad(
      femx::HostVectorView<const femx::Real> prm,
      femx::HostVectorView<femx::Real>       out,
      femx::inverse::TimeReducedProgress     progress = {}) = 0;

  virtual void        resetTiming() noexcept           = 0;
  virtual femx::Real  assemblySeconds() const noexcept = 0;
  virtual femx::Real  solveSeconds() const noexcept    = 0;
  virtual femx::Index assemblyCalls() const noexcept   = 0;
  virtual femx::Index solveCalls() const noexcept      = 0;
};

class PythonTimeProgressMonitor final
{
public:
  explicit PythonTimeProgressMonitor(pybind11::object progress)
    : progress_(std::move(progress))
  {
  }

  void progress(const char* phase,
                femx::Index step,
                femx::Index total)
  {
    pybind11::gil_scoped_acquire acquire;
    if (PyErr_CheckSignals() != 0)
    {
      throw pybind11::error_already_set();
    }
    if (progress_.is_none())
    {
      return;
    }

    pybind11::dict event;
    event["type"]  = "solve";
    event["phase"] = phase;
    event["step"]  = step;
    event["total"] = total;
    progress_(std::move(event));
  }

private:
  pybind11::object progress_;
};

void bindMesh(pybind11::module_& module);
void bindInverse(pybind11::module_& module);
void bindNavierStokes(pybind11::module_& module);
void bindState(pybind11::module_& module);
