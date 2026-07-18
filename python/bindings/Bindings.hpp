#pragma once

#include <utility>

#include <femx/inverse/TimeReducedFunctional.hpp>
#include <pybind11/pybind11.h>

class PythonTimeProgressMonitor final
  : public femx::inverse::TimeReducedProgressMonitor
{
public:
  explicit PythonTimeProgressMonitor(pybind11::object progress)
    : progress_(std::move(progress))
  {
  }

  void progress(const char* phase,
                femx::Index step,
                femx::Index total) override
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
