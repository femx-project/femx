#pragma once

#include <utility>

#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <pybind11/pybind11.h>

class PythonHostTimeIntegrator final
{
public:
  PythonHostTimeIntegrator(
      const femx::state::HostTimeResidual& res,
      femx::linalg::HostCsrLinearSolver&   solver)
    : jac_(res.graph()), integ_(res, jac_, solver, ctx_)
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
  femx::CpuContext                ctx_;
  femx::HostCsrMatrix             jac_;
  femx::state::HostTimeIntegrator integ_;
};

class PythonTimeReducedFunctional
{
public:
  virtual ~PythonTimeReducedFunctional() = default;

  virtual femx::Index numParams() const noexcept = 0;

  virtual femx::Real value(
      femx::HostConstVectorView          prm,
      femx::inverse::TimeReducedProgress progress = {}) = 0;
  virtual void grad(
      femx::HostConstVectorView          prm,
      femx::HostVectorView               out,
      femx::inverse::TimeReducedProgress progress = {}) = 0;
  virtual femx::Real valueGrad(
      femx::HostConstVectorView          prm,
      femx::HostVectorView               out,
      femx::inverse::TimeReducedProgress progress = {}) = 0;

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
