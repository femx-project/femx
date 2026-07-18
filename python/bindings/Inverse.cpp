#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "Bindings.hpp"
#include "InitialStateParameterMap.hpp"
#include "PETScInit.hpp"
#include <femx/inverse/SumTimeObjective.hpp>
#include <femx/inverse/TimeBlockRegularization.hpp>
#include <femx/inverse/TimeLeastSquaresObjective.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/inverse/TimeRegularization.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>
#ifdef FEMX_HAS_PETSC
#include <femx/opt/TaoOptimizer.hpp>
#include <femx/runtime/PETScRuntime.hpp>
#endif
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeResidual.hpp>
#include <femx/state/TimeTrajectory.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{

using femx::Array;
using femx::DenseMatrix;
using femx::HostVector;
using femx::Index;
using femx::Real;
using femx::inverse::SumTimeObjective;
using femx::inverse::TimeBlockRegularization;
using femx::inverse::TimeLeastSquaresObjective;
using femx::inverse::TimeObjective;
using femx::inverse::TimeObservationData;
using femx::inverse::TimeObservationOperator;
using femx::inverse::TimeReducedFunctional;
using femx::inverse::TimeRegularization;
using femx::linalg::LinearSolver;
using femx::linalg::MatrixOperator;
using femx::state::TimeIntegrator;
using femx::state::TimeLinearization;
using femx::state::TimeResidual;
using femx::state::TimeTrajectory;

using RealArray  = py::array_t<Real,
                               py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<Index,
                               py::array::c_style | py::array::forcecast>;

HostVector vectorFromArray(const RealArray& vals,
                           const char*      name)
{
  if (vals.ndim() != 1)
  {
    throw std::runtime_error(std::string(name) + " must be one-dimensional");
  }

  HostVector out(vals.shape(0));
  const auto data = vals.unchecked<1>();
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = data(i);
    if (!std::isfinite(out[i]))
    {
      throw std::runtime_error(std::string(name) + " must be finite");
    }
  }
  return out;
}

Array<Index> indexVectorFromArray(const IndexArray& vals,
                                  const char*       name)
{
  if (vals.ndim() != 1)
  {
    throw std::runtime_error(std::string(name) + " must be one-dimensional");
  }

  Array<Index> out(vals.shape(0));
  const auto   data = vals.unchecked<1>();
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = data(i);
  }
  return out;
}

DenseMatrix denseMatrixFromArray(const RealArray& vals,
                                 const char*      name)
{
  if (vals.ndim() != 2)
  {
    throw std::runtime_error(
        std::string(name) + " must be two-dimensional");
  }

  DenseMatrix out(vals.shape(0), vals.shape(1));
  const auto  data = vals.unchecked<2>();
  for (Index row = 0; row < out.numRows(); ++row)
  {
    for (Index col = 0; col < out.numCols(); ++col)
    {
      out(row, col) = data(row, col);
      if (!std::isfinite(out(row, col)))
      {
        throw std::runtime_error(
            std::string(name) + " must be finite");
      }
    }
  }
  return out;
}

class AffineInitialStateGradientMap final
  : public femx::inverse::InitialStateGradientMap
{
public:
  explicit AffineInitialStateGradientMap(InitialStateParameterMap& map)
    : map_(map)
  {
  }

  void apply(const HostVector&,
             const HostVector& state_grad,
             HostVector&       out) override
  {
    map_.applyTranspose(state_grad, out);
  }

private:
  InitialStateParameterMap& map_;
};

HostVector flattenedVectorFromArray(const RealArray& vals,
                                    Index            expected_size,
                                    const char*      name)
{
  if (vals.size() != expected_size)
  {
    throw std::runtime_error(
        std::string(name) + " has an inconsistent size");
  }

  HostVector  out(expected_size);
  const Real* data = vals.data();
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = data[i];
    if (!std::isfinite(out[i]))
    {
      throw std::runtime_error(std::string(name) + " must be finite");
    }
  }
  return out;
}

py::array_t<Real> vectorArray(const HostVector& vals)
{
  py::array_t<Real> out(vals.size());
  auto              data = out.mutable_unchecked<1>();
  for (Index i = 0; i < vals.size(); ++i)
  {
    data(i) = vals[i];
  }
  return out;
}

py::array_t<Index> indexArray(const Array<Index>& vals)
{
  py::array_t<Index> out(vals.size());
  auto               data = out.mutable_unchecked<1>();
  for (Index i = 0; i < vals.size(); ++i)
  {
    data(i) = vals[i];
  }
  return out;
}

TimeObservationData observationDataFromArrays(
    const RealArray&  vals,
    const py::object& times,
    const py::object& time_levels)
{
  if (vals.ndim() != 2 || vals.shape(0) == 0 || vals.shape(1) == 0)
  {
    throw std::runtime_error(
        "values must have shape (num_times, observations_per_time)");
  }
  if (!times.is_none() && !time_levels.is_none())
  {
    throw std::runtime_error(
        "provide either times or time_levels, not both");
  }

  TimeObservationData out(vals.shape(0), vals.shape(1));
  const auto          data = vals.unchecked<2>();
  for (Index row = 0; row < out.numTimeLevels(); ++row)
  {
    auto out_row = out[row];
    for (Index column = 0; column < out.numObservations(); ++column)
    {
      const Real value = data(row, column);
      if (!std::isfinite(value))
      {
        throw std::runtime_error("observation values must be finite");
      }
      out_row[column] = value;
    }
  }

  if (!times.is_none())
  {
    const RealArray time_vals = RealArray::ensure(times);
    if (!time_vals)
    {
      throw std::runtime_error("times must be real-valued");
    }
    out.setTimeValues(vectorFromArray(time_vals, "times"));
  }
  else if (!time_levels.is_none())
  {
    const IndexArray level_vals = IndexArray::ensure(time_levels);
    if (!level_vals || level_vals.ndim() != 1)
    {
      throw std::runtime_error("time_levels must be one-dimensional");
    }
    Array<Index> levels(level_vals.shape(0));
    const auto   level_data = level_vals.unchecked<1>();
    for (Index i = 0; i < levels.size(); ++i)
    {
      levels[i] = level_data(i);
    }
    out.setTimeLevels(std::move(levels));
  }

  return out;
}

py::array_t<Real> observationDataArray(const TimeObservationData& vals)
{
  py::array_t<Real> out(
      {vals.numTimeLevels(), vals.numObservations()});
  auto data = out.mutable_unchecked<2>();
  for (Index row = 0; row < vals.numTimeLevels(); ++row)
  {
    const auto src = vals[row];
    for (Index column = 0; column < vals.numObservations(); ++column)
    {
      data(row, column) = src[column];
    }
  }
  return out;
}

std::unique_ptr<TimeLeastSquaresObjective> timeLeastSquaresFromArrays(
    const TimeObservationOperator& observation,
    TimeObservationData            data,
    const RealArray&               level_weights,
    const RealArray&               obs_weights,
    Real                           dt,
    Real                           time_offset)
{
  HostVector levels  = vectorFromArray(level_weights, "level_weights");
  HostVector entries = flattenedVectorFromArray(
      obs_weights,
      data.numTimeLevels() * data.numObservations(),
      "obs_weights");
  return std::make_unique<TimeLeastSquaresObjective>(
      observation,
      std::move(data),
      std::move(levels),
      std::move(entries),
      dt,
      time_offset);
}

std::unique_ptr<TimeRegularization> timeRegularizationFromArray(
    Index             num_steps,
    Index             num_states,
    Index             num_levels,
    Index             block_size,
    Real              difference_weight,
    Real              value_weight,
    const py::object& reference)
{
  HostVector vals;
  if (!reference.is_none())
  {
    const RealArray array = RealArray::ensure(reference);
    if (!array)
    {
      throw std::runtime_error("reference must be real-valued");
    }
    vals = flattenedVectorFromArray(
        array, num_levels * block_size, "reference");
  }
  return std::make_unique<TimeRegularization>(
      num_steps,
      num_states,
      num_levels,
      block_size,
      difference_weight,
      value_weight,
      vals);
}

std::unique_ptr<TimeBlockRegularization> timeBlockRegularizationFromArrays(
    Index             num_steps,
    Index             num_states,
    Index             num_levels,
    Index             block_size,
    const IndexArray& rows,
    const IndexArray& cols,
    const RealArray&  vals,
    Real              weight,
    const py::object& reference)
{
  HostVector ref;
  if (!reference.is_none())
  {
    const RealArray array = RealArray::ensure(reference);
    if (!array)
    {
      throw std::runtime_error("reference must be real-valued");
    }
    ref = flattenedVectorFromArray(
        array, num_levels * block_size, "reference");
  }
  return std::make_unique<TimeBlockRegularization>(
      num_steps,
      num_states,
      num_levels,
      block_size,
      indexVectorFromArray(rows, "rows"),
      indexVectorFromArray(cols, "cols"),
      vectorFromArray(vals, "vals"),
      weight,
      ref);
}

class PythonTimeProgressMonitor final
  : public femx::inverse::TimeReducedProgressMonitor
{
public:
  explicit PythonTimeProgressMonitor(py::object progress)
    : progress_(std::move(progress))
  {
  }

  void progress(const char* phase, Index step, Index total) override
  {
    py::gil_scoped_acquire acquire;
    if (PyErr_CheckSignals() != 0)
    {
      throw py::error_already_set();
    }
    if (progress_.is_none())
    {
      return;
    }

    py::dict event;
    event["type"]  = "solve";
    event["phase"] = phase;
    event["step"]  = step;
    event["total"] = total;
    progress_(std::move(event));
  }

private:
  py::object progress_;
};

#ifdef FEMX_HAS_PETSC
HostVector boundVectorFromArray(const RealArray& vals,
                                Index            size,
                                const char*      name)
{
  if (vals.ndim() != 1 || vals.shape(0) != size)
  {
    throw std::runtime_error(
        std::string(name) + " must contain one value per parameter");
  }

  HostVector out(size);
  const auto data = vals.unchecked<1>();
  for (Index i = 0; i < size; ++i)
  {
    out[i] = data(i);
    if (std::isnan(out[i]))
    {
      throw std::runtime_error(std::string(name) + " must not contain NaN");
    }
  }
  return out;
}

const char* taoReason(TaoConvergedReason reason)
{
  switch (reason)
  {
  case TAO_CONVERGED_GATOL:
    return "TAO converged: absolute gradient tolerance";
  case TAO_CONVERGED_GRTOL:
    return "TAO converged: relative gradient tolerance";
  case TAO_CONVERGED_GTTOL:
    return "TAO converged: gradient reduction tolerance";
  case TAO_CONVERGED_STEPTOL:
    return "TAO converged: step tolerance";
  case TAO_CONVERGED_MINF:
    return "TAO converged: objective lower bound";
  case TAO_CONVERGED_USER:
    return "TAO converged: user condition";
  case TAO_DIVERGED_MAXITS:
    return "TAO stopped: maximum iterations reached";
  case TAO_DIVERGED_NAN:
    return "TAO failed: non-finite value";
  case TAO_DIVERGED_MAXFCN:
    return "TAO stopped: maximum function evaluations reached";
  case TAO_DIVERGED_LS_FAILURE:
    return "TAO failed: line search";
  case TAO_DIVERGED_TR_REDUCTION:
    return "TAO failed: trust-region reduction";
  case TAO_DIVERGED_USER:
    return "TAO stopped: user condition";
  case TAO_CONTINUE_ITERATING:
    return "TAO stopped without a convergence reason";
  }
  return "TAO stopped with an unknown reason";
}

py::dict taoSolve(TimeReducedFunctional& functional,
                  const RealArray&       initial,
                  const py::object&      lower,
                  const py::object&      upper,
                  Index                  max_itrs,
                  Real                   grad_tol,
                  const py::object&      progress)
{
  if (max_itrs <= 0)
  {
    throw std::runtime_error("max_itrs must be positive");
  }
  if (!std::isfinite(grad_tol) || grad_tol < 0.0)
  {
    throw std::runtime_error("grad_tol must be finite and nonnegative");
  }
  if (lower.is_none() != upper.is_none())
  {
    throw std::runtime_error("lower and upper must be provided together");
  }

  const HostVector init      = vectorFromArray(initial, "init_param");
  Index            num_evals = 0;

  struct CalculationInterrupted
  {
  };

  class InterruptMonitor final
    : public femx::opt::TaoProgressMonitor,
      public femx::inverse::TimeReducedProgressMonitor
  {
  public:
    InterruptMonitor(py::object progress, Index max_itrs)
      : progress_(std::move(progress)),
        max_itrs_(max_itrs)
    {
    }

    void observe(const femx::opt::TaoIterationInfo& info,
                 const HostVector&                  param) override
    {
      py::gil_scoped_acquire acquire;
      py::dict               event;
      event["type"]           = "optimizer";
      event["iteration"]      = info.its;
      event["max_iterations"] = max_itrs_;
      event["objective"]      = info.value;
      event["gradient_norm"]  = info.grad_norm;
      event["param"]          = vectorArray(param);
      notify(std::move(event), false);
    }

    void progress(const char* phase, Index step, Index total) override
    {
      py::gil_scoped_acquire acquire;
      py::dict               event;
      event["type"]  = "solve";
      event["phase"] = phase;
      event["step"]  = step;
      event["total"] = total;
      notify(std::move(event), true);
    }

    bool interrupted() const
    {
      return interrupted_;
    }

  private:
    void notify(py::dict event, bool stop)
    {
      if (!interrupted_ && PyErr_CheckSignals() != 0)
      {
        interrupted_ = true;
      }
      if (!interrupted_ && !progress_.is_none())
      {
        try
        {
          progress_(std::move(event));
        }
        catch (py::error_already_set& error)
        {
          error.restore();
          interrupted_ = true;
        }
      }
      if (interrupted_ && stop)
      {
        throw CalculationInterrupted{};
      }
    }

    py::object progress_;
    Index      max_itrs_;
    bool       interrupted_{false};
  } interrupt_monitor(progress, max_itrs);

  femx::python::initializePETSc();
  femx::opt::TaoOptimizer tao(
      [&functional]()
      { return functional.numParams(); },
      [&functional, &num_evals](const HostVector& param,
                                HostVector&       grad)
      {
        ++num_evals;
        try
        {
          return functional.valueGrad(param, grad);
        }
        catch (const CalculationInterrupted&)
        {
          grad.resize(functional.numParams());
          return 0.0;
        }
      },
      PETSC_COMM_WORLD);
  tao.opts().abs_tol            = grad_tol;
  tao.opts().rel_tol            = 0.0;
  tao.opts().grad_reduction_tol = 0.0;
  tao.opts().max_its            = max_itrs;
  tao.setMonitor(&interrupt_monitor);
  functional.setMonitor(&interrupt_monitor);

  if (!lower.is_none())
  {
    const RealArray lower_vals = RealArray::ensure(lower);
    const RealArray upper_vals = RealArray::ensure(upper);
    if (!lower_vals || !upper_vals)
    {
      throw std::runtime_error("bounds must be real-valued");
    }
    tao.setBounds(
        boundVectorFromArray(lower_vals, functional.numParams(), "lower"),
        boundVectorFromArray(upper_vals, functional.numParams(), "upper"));
  }

  femx::opt::TaoResult result;
  PetscErrorCode       ierr = PETSC_SUCCESS;
  {
    py::gil_scoped_release release;
    ierr = tao.solve(init, result);
  }
  functional.clearMonitor();
  if (interrupt_monitor.interrupted())
  {
    if (PyErr_Occurred() == nullptr)
    {
      PyErr_SetNone(PyExc_KeyboardInterrupt);
    }
    throw py::error_already_set();
  }
  femx::runtime::checkPetsc(ierr, "TAO solve");

  py::dict out;
  out["param"]     = vectorArray(result.prm);
  out["grad"]      = vectorArray(result.grad);
  out["obj"]       = result.value;
  out["num_iter"]  = result.its;
  out["num_fun"]   = num_evals;
  out["num_grad"]  = num_evals;
  out["converged"] = result.converged();
  out["status"]    = static_cast<int>(result.reason);
  out["msg"]       = taoReason(result.reason);
  return out;
}
#endif

} // namespace

void bindInverse(py::module_& module)
{
  py::class_<TimeObservationOperator>(module, "TimeObservationOperator")
      .def_property_readonly("num_steps",
                             &TimeObservationOperator::numSteps)
      .def_property_readonly("num_states",
                             &TimeObservationOperator::numStates)
      .def_property_readonly("num_param",
                             &TimeObservationOperator::numParams)
      .def_property_readonly("num_obs",
                             &TimeObservationOperator::numObservations);

  py::class_<TimeObservationData>(module, "TimeObservationData")
      .def(py::init(&observationDataFromArrays),
           py::arg("values"),
           py::kw_only(),
           py::arg("times")       = py::none(),
           py::arg("time_levels") = py::none())
      .def_property_readonly("num_levels",
                             &TimeObservationData::numTimeLevels)
      .def_property_readonly("num_obs",
                             &TimeObservationData::numObservations)
      .def_property_readonly("values", &observationDataArray)
      .def_property_readonly(
          "times",
          [](const TimeObservationData& data) -> py::object
          {
            if (!data.hasTimeValues())
            {
              return py::none();
            }
            return vectorArray(data.timeValues());
          })
      .def_property_readonly(
          "time_levels",
          [](const TimeObservationData& data) -> py::object
          {
            if (!data.hasTimeLevels())
            {
              return py::none();
            }
            return indexArray(data.timeLevels());
          });

  py::class_<TimeObjective>(module, "TimeObjective")
      .def_property_readonly("num_steps", &TimeObjective::numSteps)
      .def_property_readonly("num_states", &TimeObjective::numStates)
      .def_property_readonly("num_param", &TimeObjective::numParams)
      .def(
          "value",
          [](const TimeObjective&  objective,
             const TimeTrajectory& trajectory,
             const RealArray&      parameters)
          {
            const HostVector vals =
                vectorFromArray(parameters, "parameters");
            py::gil_scoped_release release;
            return objective.value(trajectory, vals);
          },
          py::arg("trajectory"),
          py::arg("param"))
      .def(
          "state_grad",
          [](const TimeObjective&  objective,
             Index                 level,
             const TimeTrajectory& trajectory,
             const RealArray&      parameters)
          {
            const HostVector vals =
                vectorFromArray(parameters, "parameters");
            HostVector out;
            {
              py::gil_scoped_release release;
              objective.stateGrad(level, trajectory, vals, out);
            }
            return vectorArray(out);
          },
          py::arg("level"),
          py::arg("trajectory"),
          py::arg("param"))
      .def(
          "param_grad",
          [](const TimeObjective&  objective,
             const TimeTrajectory& trajectory,
             const RealArray&      parameters)
          {
            const HostVector vals =
                vectorFromArray(parameters, "parameters");
            HostVector out;
            {
              py::gil_scoped_release release;
              objective.paramGrad(trajectory, vals, out);
            }
            return vectorArray(out);
          },
          py::arg("trajectory"),
          py::arg("param"));

  py::class_<TimeLeastSquaresObjective, TimeObjective>(
      module, "TimeLeastSquaresObjective")
      .def(py::init(&timeLeastSquaresFromArrays),
           py::arg("operator"),
           py::arg("data"),
           py::arg("level_weights"),
           py::arg("obs_weights"),
           py::arg("dt"),
           py::arg("time_offset") = 0.0,
           py::keep_alive<1, 2>());

  py::class_<TimeRegularization, TimeObjective>(
      module, "TimeRegularization")
      .def(py::init(&timeRegularizationFromArray),
           py::arg("num_steps"),
           py::arg("num_states"),
           py::arg("num_levels"),
           py::arg("block_size"),
           py::arg("difference_weight"),
           py::arg("value_weight") = 0.0,
           py::arg("reference")    = py::none());

  py::class_<TimeBlockRegularization, TimeObjective>(
      module, "TimeBlockRegularization")
      .def(py::init(&timeBlockRegularizationFromArrays),
           py::arg("num_steps"),
           py::arg("num_states"),
           py::arg("num_levels"),
           py::arg("block_size"),
           py::arg("rows"),
           py::arg("cols"),
           py::arg("vals"),
           py::arg("weight"),
           py::arg("reference") = py::none());

  py::class_<SumTimeObjective, TimeObjective>(module, "SumTimeObjective")
      .def(py::init<Index, Index, Index>(),
           py::arg("num_steps"),
           py::arg("num_states"),
           py::arg("num_param"))
      .def(
          "add",
          [](SumTimeObjective&    objective,
             const TimeObjective& term) -> SumTimeObjective&
          {
            return objective.add(term);
          },
          py::arg("term"),
          py::return_value_policy::reference_internal,
          py::keep_alive<1, 2>());

  py::class_<TimeReducedFunctional>(module, "TimeReducedFunctional")
      .def(py::init<TimeIntegrator&,
                    const TimeResidual&,
                    TimeLinearization&,
                    MatrixOperator&,
                    MatrixOperator&,
                    LinearSolver&,
                    const TimeObjective&>(),
           py::arg("integrator"),
           py::arg("problem"),
           py::arg("linearization"),
           py::arg("next_state_matrix"),
           py::arg("history_matrix"),
           py::arg("adjoint_solver"),
           py::arg("objective"),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>(),
           py::keep_alive<1, 4>(),
           py::keep_alive<1, 5>(),
           py::keep_alive<1, 6>(),
           py::keep_alive<1, 7>(),
           py::keep_alive<1, 8>())
      .def_property_readonly("num_param",
                             &TimeReducedFunctional::numParams)
      .def_property_readonly("assembly_seconds",
                             &TimeReducedFunctional::assemblySeconds)
      .def_property_readonly("solve_seconds",
                             &TimeReducedFunctional::solveSeconds)
      .def_property_readonly("assembly_calls",
                             &TimeReducedFunctional::assemblyCalls)
      .def_property_readonly("solve_calls",
                             &TimeReducedFunctional::solveCalls)
      .def("reset_timing", &TimeReducedFunctional::resetTiming)
      .def(
          "set_initial_state_param_jac_t",
          [](TimeReducedFunctional&         functional,
             AffineInitialStateGradientMap& mapping)
          {
            functional.setInitialStateParamJacT(&mapping);
          },
          py::arg("mapping"),
          py::keep_alive<1, 2>())
      .def("clear_initial_state_param_jac_t",
           &TimeReducedFunctional::clearInitialStateParamJacT)
      .def(
          "value",
          [](TimeReducedFunctional& functional,
             const RealArray&       parameters)
          {
            const HostVector vals =
                vectorFromArray(parameters, "parameters");
            py::gil_scoped_release release;
            return functional.value(vals);
          },
          py::arg("param"))
      .def(
          "grad",
          [](TimeReducedFunctional& functional,
             const RealArray&       parameters)
          {
            const HostVector vals =
                vectorFromArray(parameters, "parameters");
            HostVector out;
            {
              py::gil_scoped_release release;
              functional.grad(vals, out);
            }
            return vectorArray(out);
          },
          py::arg("param"))
      .def(
          "value_grad",
          [](TimeReducedFunctional& functional,
             const RealArray&       parameters,
             const py::object&      progress)
          {
            const HostVector vals =
                vectorFromArray(parameters, "parameters");
            PythonTimeProgressMonitor monitor(progress);
            HostVector                out;
            Real                      value = 0.0;
            functional.setMonitor(&monitor);
            try
            {
              py::gil_scoped_release release;
              value = functional.valueGrad(vals, out);
            }
            catch (...)
            {
              functional.clearMonitor();
              throw;
            }
            functional.clearMonitor();
            return py::make_tuple(value, vectorArray(out));
          },
          py::arg("param"),
          py::arg("progress") = py::none());

  py::class_<AffineInitialStateGradientMap>(
      module, "_AffineInitialStateGradientMap")
      .def(py::init<InitialStateParameterMap&>(),
           py::arg("map"),
           py::keep_alive<1, 2>());

#ifdef FEMX_HAS_PETSC
  module.def("_tao_solve",
             &taoSolve,
             py::arg("functional"),
             py::arg("init_param"),
             py::kw_only(),
             py::arg("lower")    = py::none(),
             py::arg("upper")    = py::none(),
             py::arg("max_itrs") = 100,
             py::arg("grad_tol") = 1.0e-8,
             py::arg("progress") = py::none());
#endif
}
