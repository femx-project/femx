#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "Bindings.hpp"
#include "PETScInit.hpp"
#include <femx/fem/ControlMap.hpp>
#include <femx/linalg/Dense.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#ifdef FEMX_HAS_PETSC
#include <femx/runtime/PETScRuntime.hpp>
#endif
#ifdef FEMX_HAS_RESOLVE
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#endif
#include <femx/state/EnsembleBasis.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeResidual.hpp>
#include <femx/state/TimeTrajectory.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{

using femx::DenseMatrix;
using femx::HostConstVectorView;
using femx::HostVector;
using femx::HostVectorView;
using femx::Index;
using femx::Real;
using femx::fem::HostInitialStateMap;
using femx::linalg::DenseLinearSolver;
using femx::linalg::HostCsrLinearSolver;
#ifdef FEMX_HAS_PETSC
#endif
#ifdef FEMX_HAS_RESOLVE
using femx::linalg::ReSolveLinearSolver;
using femx::linalg::ReSolveOptions;
#endif
using femx::state::EnsembleBasis;
using femx::state::HostTimeContext;
using femx::state::HostTimeHistoryView;
using femx::state::HostTimeIntegrator;
using femx::state::TimeDims;
using femx::state::TimeStepStateContext;
using TimeResidual = femx::state::HostTimeResidual;
using femx::state::TimeTrajectory;
using femx::state::VariableBlock;

using RealArray = py::array_t<Real,
                              py::array::c_style | py::array::forcecast>;

HostVector vectorFromArray(const RealArray& vals,
                           const char*      name)
{
  if (vals.ndim() != 1)
  {
    throw std::runtime_error(std::string(name) + " must be one-dimensional");
  }

  HostVector out(static_cast<Index>(vals.shape(0)));
  const auto data = vals.unchecked<1>();
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = data(i);
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

py::array_t<Real> vectorArray(HostConstVectorView vals)
{
  py::array_t<Real> out(vals.size());
  auto              data = out.mutable_unchecked<1>();
  for (Index i = 0; i < vals.size(); ++i)
  {
    data(i) = vals[i];
  }
  return out;
}

class PythonTimeObserver
{
public:
  explicit PythonTimeObserver(py::object progress)
    : progress_(std::move(progress))
  {
  }

  bool operator()(const TimeStepStateContext& ctx)
  {
    py::gil_scoped_acquire acquire;
    checkSignals();
    if (ctx.level == 0)
    {
      return false;
    }
    py::dict event;
    event["type"]                 = "solve";
    event["phase"]                = "forward";
    event["step"]                 = ctx.level;
    event["total"]                = ctx.total_steps;
    event["assembly_seconds"]     = ctx.assm_sec;
    event["linear_solve_seconds"] = ctx.lin_solve_sec;
    progress_(std::move(event));
    return false;
  }

private:
  static void checkSignals()
  {
    if (PyErr_CheckSignals() != 0)
    {
      throw py::error_already_set();
    }
  }

  py::object progress_;
};

DenseMatrix denseMatrixFromArray(const RealArray& vals,
                                 const char*      name)
{
  if (vals.ndim() != 2)
  {
    throw std::runtime_error(
        std::string(name) + " must be two-dimensional");
  }

  const Index rows = static_cast<Index>(vals.shape(0));
  const Index cols = static_cast<Index>(vals.shape(1));
  DenseMatrix out(rows, cols);
  const auto  data = vals.unchecked<2>();
  for (Index row = 0; row < rows; ++row)
  {
    for (Index col = 0; col < cols; ++col)
    {
      out(row, col) = data(row, col);
    }
  }
  return out;
}

py::array_t<Real> denseMatrixArray(const DenseMatrix& vals)
{
  py::array_t<Real> out({vals.rows(), vals.cols()});
  auto              data = out.mutable_unchecked<2>();
  for (Index row = 0; row < vals.rows(); ++row)
  {
    for (Index col = 0; col < vals.cols(); ++col)
    {
      data(row, col) = vals(row, col);
    }
  }
  return out;
}

void checkFinite(const HostVector& vals, const char* name)
{
  for (Real value : vals)
  {
    if (!std::isfinite(value))
    {
      throw std::runtime_error(std::string(name) + " must be finite");
    }
  }
}

void checkFinite(const DenseMatrix& vals, const char* name)
{
  for (Index row = 0; row < vals.rows(); ++row)
  {
    for (Index col = 0; col < vals.cols(); ++col)
    {
      if (!std::isfinite(vals(row, col)))
      {
        throw std::runtime_error(std::string(name) + " must be finite");
      }
    }
  }
}

EnsembleBasis ensembleBasisFromArrays(const RealArray& mean,
                                      const RealArray& perturbations)
{
  HostVector  mean_vals = vectorFromArray(mean, "mean");
  DenseMatrix perturb_vals =
      denseMatrixFromArray(perturbations, "perturbations");
  if (mean_vals.empty())
  {
    throw std::runtime_error("mean must not be empty");
  }
  if (perturb_vals.rows() != mean_vals.size()
      || perturb_vals.cols() <= 0)
  {
    throw std::runtime_error(
        "perturbations must have shape (value_size, num_coefficients)");
  }
  checkFinite(mean_vals, "mean");
  checkFinite(perturb_vals, "perturbations");
  return EnsembleBasis(
      std::move(mean_vals), std::move(perturb_vals));
}

void copyArray(const py::handle& value,
               HostVector&       out,
               const char*       name)
{
  const RealArray vals = RealArray::ensure(value);
  if (!vals)
  {
    throw std::runtime_error(std::string(name) + " must be a real NumPy array");
  }
  out = vectorFromArray(vals, name);
}

py::array_t<Real> historyArray(const HostTimeHistoryView& history)
{
  py::array_t<Real> out({history.count(), history.stateSize()});
  auto              data = out.mutable_unchecked<2>();
  for (Index lag = 0; lag < history.count(); ++lag)
  {
    const auto state = history.state(lag);
    for (Index i = 0; i < history.stateSize(); ++i)
    {
      data(lag, i) = state[i];
    }
  }
  return out;
}

py::dict ctxData(const HostTimeContext& ctx)
{
  py::dict out;
  out["step"]       = ctx.step;
  out["next_state"] = vectorArray(ctx.nxt);
  out["parameters"] = vectorArray(ctx.prm);
  out["history"]    = historyArray(ctx.hist);
  return out;
}

class PyTimeResidual : public TimeResidual
{
public:
  using TimeResidual::TimeResidual;

  TimeDims dims() const override
  {
    PYBIND11_OVERRIDE_PURE(TimeDims, TimeResidual, dims);
  }

  const femx::HostCsrGraph& hostGraph() const override
  {
    updateGraph();
    return graph_;
  }

  const femx::HostCsrGraph& graph() const override
  {
    updateGraph();
    return graph_;
  }

  void initialState(HostConstVectorView prm,
                    HostVector&         out,
                    femx::CpuContext&) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override = py::get_override(this, "initial_state");
    if (!override)
    {
      resizeOrZero(out, dims().num_states);
      return;
    }
    copyArray(override(vectorArray(prm)), out, "initial state");
  }

  void addInitialStateJacobianTranspose(
      HostConstVectorView state_grad,
      HostVectorView      out,
      femx::CpuContext&   ctx) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override =
        py::get_override(this, "add_initial_state_jacobian_transpose");
    if (!override)
    {
      TimeResidual::addInitialStateJacobianTranspose(
          state_grad, out, ctx);
      return;
    }
    HostVector grad;
    copyArray(override(vectorArray(state_grad)),
              grad,
              "initial-state transpose result");
    if (grad.size() != out.size())
    {
      throw std::runtime_error(
          "initial-state transpose result has invalid size");
    }
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] += grad[i];
    }
  }

  void res(const HostTimeContext& ctx,
           HostVector&            out,
           femx::CpuContext&) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override = py::get_override(this, "residual");
    if (!override)
    {
      throw std::runtime_error("TimeResidual.residual() is not implemented");
    }
    copyArray(override(ctxData(ctx)), out, "residual result");
  }

  void applyJacT(const HostTimeContext& ctx,
                 VariableBlock          wrt,
                 HostConstVectorView    adj,
                 HostVector&            out,
                 femx::CpuContext&) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override =
        py::get_override(this, "apply_jacobian_transpose");
    if (!override)
    {
      throw std::runtime_error(
          "TimeResidual.apply_jacobian_transpose() is not implemented");
    }
    copyArray(override(ctxData(ctx), wrt, vectorArray(adj)),
              out,
              "transpose Jacobian result");
  }

  void assembleNext(const HostTimeContext& ctx,
                    HostVector&            res_out,
                    femx::HostCsrMatrix&   jac,
                    femx::CpuContext&      cpu) const override
  {
    res(ctx, res_out, cpu);
    updateGraph();
    if (jac.graph().layoutId() != graph_.layoutId())
    {
      throw std::runtime_error(
          "Python TimeResidual Jacobian uses an incompatible graph");
    }

    py::gil_scoped_acquire gil;
    const py::function     override =
        py::get_override(this, "assemble_next");
    if (!override)
    {
      throw std::runtime_error(
          "TimeResidual.assemble_next() is not implemented");
    }

    const py::object value = override(ctxData(ctx));
    const RealArray  mat   = RealArray::ensure(value);
    if (!mat || mat.ndim() != 2)
    {
      throw std::runtime_error(
          "TimeResidual.assemble_next() must return a two-dimensional array");
    }

    const TimeDims dims = this->dims();
    const Index    rows = static_cast<Index>(mat.shape(0));
    const Index    cols = static_cast<Index>(mat.shape(1));
    if (rows != dims.num_res || cols != dims.num_states)
    {
      throw std::runtime_error(
          "TimeResidual.assemble_next() returned an array with invalid shape");
    }

    jac.setZero();
    const auto data = mat.unchecked<2>();
    for (Index row = 0; row < rows; ++row)
    {
      for (Index k = jac.rowPtrData()[row];
           k < jac.rowPtrData()[row + 1];
           ++k)
      {
        jac.valsData()[k] = data(row, jac.colIndData()[k]);
      }
    }
  }

  void prepareLinearSolve(const HostTimeContext& ctx,
                          femx::HostCsrMatrix&   jac,
                          HostVector&            rhs,
                          femx::CpuContext&) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override =
        py::get_override(this, "prepare_linear_solve");
    if (!override)
    {
      return;
    }

    DenseMatrix dense(jac.rows(), jac.cols());
    for (Index row = 0; row < jac.rows(); ++row)
    {
      for (Index k = jac.rowPtrData()[row];
           k < jac.rowPtrData()[row + 1];
           ++k)
      {
        dense(row, jac.colIndData()[k]) = jac.valsData()[k];
      }
    }
    py::array_t<Real> jac_array = denseMatrixArray(dense);
    py::array_t<Real> rhs_array = vectorArray(rhs);
    const py::object  result    = override(
        ctxData(ctx),
        jac_array,
        rhs_array);
    if (result.is_none())
    {
      copyArray(rhs_array, rhs, "prepared right-hand side");
    }
    else
    {
      copyArray(result, rhs, "prepared right-hand side");
    }
    if (jac_array.ndim() != 2 || jac_array.shape(0) != jac.rows()
        || jac_array.shape(1) != jac.cols())
    {
      throw std::runtime_error(
          "prepared Jacobian has an inconsistent shape");
    }
    const auto jac_vals = jac_array.unchecked<2>();
    for (Index row = 0; row < jac.rows(); ++row)
    {
      for (Index k = jac.rowPtrData()[row];
           k < jac.rowPtrData()[row + 1];
           ++k)
      {
        jac.valsData()[k] = jac_vals(row, jac.colIndData()[k]);
      }
    }
  }

private:
  void updateGraph() const
  {
    const TimeDims dim = dims();
    if (graph_.rows() == dim.num_res && graph_.cols() == dim.num_states)
    {
      return;
    }
    femx::HostIndexVector row_ptr(dim.num_res + 1);
    femx::HostIndexVector col_ind(dim.num_res * dim.num_states);
    for (Index row = 0; row <= dim.num_res; ++row)
    {
      row_ptr[row] = row * dim.num_states;
    }
    for (Index row = 0; row < dim.num_res; ++row)
    {
      for (Index col = 0; col < dim.num_states; ++col)
      {
        col_ind[row * dim.num_states + col] = col;
      }
    }
    graph_ = femx::HostCsrGraph(
        dim.num_res, dim.num_states, std::move(row_ptr), std::move(col_ind));
  }

  mutable femx::HostCsrGraph graph_;
};

py::array trajectoryValues(TimeTrajectory& trajectory)
{
  const py::ssize_t levels = trajectory.numTimeLevels();
  const py::ssize_t states = trajectory.numStates();
  return py::array_t<Real>(
      {levels, states},
      {states * static_cast<py::ssize_t>(sizeof(Real)),
       static_cast<py::ssize_t>(sizeof(Real))},
      trajectory.data(),
      py::cast(&trajectory, py::return_value_policy::reference));
}

py::array trajectoryLevel(TimeTrajectory& trajectory, Index level)
{
  if (level < 0)
  {
    level += trajectory.numTimeLevels();
  }
  auto vals = trajectory.level(level);
  return py::array_t<Real>(
      {static_cast<py::ssize_t>(vals.size())},
      {static_cast<py::ssize_t>(sizeof(Real))},
      vals.data(),
      py::cast(&trajectory, py::return_value_policy::reference));
}

} // namespace

void bindState(py::module_& module)
{
  py::class_<EnsembleBasis>(module, "EnsembleBasis")
      .def(py::init(&ensembleBasisFromArrays),
           py::arg("mean"),
           py::arg("perturbations"))
      .def_property_readonly("value_size",
                             &EnsembleBasis::numPhysicalParams)
      .def_property_readonly("num_physical_parameters",
                             &EnsembleBasis::numPhysicalParams)
      .def_property_readonly("num_coefficients",
                             &EnsembleBasis::numCoefficients)
      .def_property_readonly(
          "mean",
          [](const EnsembleBasis& basis)
          {
            return vectorArray(basis.mean());
          })
      .def_property_readonly(
          "perturbations",
          [](const EnsembleBasis& basis)
          {
            return denseMatrixArray(basis.perturbations());
          })
      .def(
          "evaluate",
          [](const EnsembleBasis& basis, const RealArray& coefficients)
          {
            HostVector coeffs =
                vectorFromArray(coefficients, "coefficients");
            checkFinite(coeffs, "coefficients");
            HostVector out;
            basis.apply(coeffs, out);
            return vectorArray(out);
          },
          py::arg("coefficients"))
      .def(
          "apply_transpose",
          [](const EnsembleBasis& basis, const RealArray& vals)
          {
            HostVector phys = vectorFromArray(vals, "values");
            checkFinite(phys, "values");
            HostVector out;
            basis.applyT(phys, out);
            return vectorArray(out);
          },
          py::arg("values"))
      .def(
          "reset",
          [](EnsembleBasis&   basis,
             const RealArray& mean,
             const RealArray& perturbations)
          {
            basis = ensembleBasisFromArrays(mean, perturbations);
          },
          py::arg("mean"),
          py::arg("perturbations"));

  py::class_<HostCsrLinearSolver>(module, "HostCsrLinearSolver");
  py::class_<DenseLinearSolver, HostCsrLinearSolver>(module,
                                                     "DenseLinearSolver")
      .def(py::init<Real>(), py::arg("pivot_tolerance") = 1.0e-14);

#ifdef FEMX_HAS_RESOLVE
  py::class_<ReSolveOptions>(module, "_ReSolveOptions")
      .def(py::init<>())
      .def_readwrite("factor", &ReSolveOptions::factor)
      .def_readwrite("refactor", &ReSolveOptions::refactor)
      .def_readwrite("solve", &ReSolveOptions::solve)
      .def_readwrite("precond", &ReSolveOptions::precond)
      .def_readwrite("ir", &ReSolveOptions::ir)
      .def_readwrite("gram_schmidt", &ReSolveOptions::gram_schmidt)
      .def_readwrite("sketching", &ReSolveOptions::sketching)
      .def_readwrite("pc_side", &ReSolveOptions::pc_side)
      .def_readwrite("max_its", &ReSolveOptions::max_its)
      .def_readwrite("restart", &ReSolveOptions::restart)
      .def_readwrite("rtol", &ReSolveOptions::rtol)
      .def_readwrite("flexible", &ReSolveOptions::flexible);

  py::class_<ReSolveLinearSolver, HostCsrLinearSolver>(
      module, "_ReSolveLinearSolver")
      .def(py::init<>())
      .def(py::init<ReSolveOptions>(), py::arg("options"));
#endif

#ifdef FEMX_HAS_PETSC
  module.def(
      "_petsc_world_rank",
      []()
      {
        femx::python::initializePETSc();
        return femx::runtime::commRank(PETSC_COMM_WORLD);
      });
  module.def(
      "_petsc_world_size",
      []()
      {
        femx::python::initializePETSc();
        return femx::runtime::commSize(PETSC_COMM_WORLD);
      });
  module.def(
      "_petsc_world_barrier",
      []()
      {
        femx::python::initializePETSc();
        py::gil_scoped_release release;
        femx::runtime::checkPetsc(
            MPI_Barrier(PETSC_COMM_WORLD), "MPI_Barrier");
      });
  module.def(
      "_petsc_finalize",
      []()
      {
        PetscBool initialized = PETSC_FALSE;
        femx::runtime::checkPetsc(
            PetscInitialized(&initialized), "PetscInitialized");
        PetscBool finalized = PETSC_FALSE;
        femx::runtime::checkPetsc(
            PetscFinalized(&finalized), "PetscFinalized");
        if (initialized == PETSC_TRUE && finalized != PETSC_TRUE)
        {
          py::gil_scoped_release release;
          femx::runtime::checkPetsc(PetscFinalize(), "PetscFinalize");
        }
      });

#endif

  py::class_<TimeDims>(module, "TimeDims")
      .def(py::init<>())
      .def_readwrite("num_steps", &TimeDims::num_steps)
      .def_readwrite("num_states", &TimeDims::num_states)
      .def_readwrite("num_param", &TimeDims::num_param)
      .def_readwrite("num_res", &TimeDims::num_res)
      .def_readwrite("num_hist", &TimeDims::num_hist);

  py::class_<VariableBlock>(module, "VariableBlock")
      .def_static("history", &VariableBlock::hist, py::arg("lag"))
      .def_property_readonly("is_history_state",
                             &VariableBlock::isHistoryState)
      .def_property_readonly("is_parameter", &VariableBlock::isParam)
      .def_property_readonly("history_lag",
                             &VariableBlock::historyLag);

  py::class_<TimeResidual, PyTimeResidual>(module, "TimeResidual")
      .def(py::init<>())
      .def("dims", &TimeResidual::dims);

  py::class_<TimeTrajectory>(module,
                             "TimeTrajectory",
                             py::buffer_protocol())
      .def(py::init<>())
      .def(py::init<Index, Index>(),
           py::arg("num_steps"),
           py::arg("num_states"))
      .def_property_readonly("num_steps", &TimeTrajectory::numSteps)
      .def_property_readonly("num_time_levels",
                             &TimeTrajectory::numTimeLevels)
      .def_property_readonly("num_states", &TimeTrajectory::numStates)
      .def_property_readonly("shape",
                             [](const TimeTrajectory& trajectory)
                             {
                               return py::make_tuple(
                                   trajectory.numTimeLevels(),
                                   trajectory.numStates());
                             })
      .def_property_readonly("values", &trajectoryValues, py::return_value_policy::reference_internal)
      .def("__len__", &TimeTrajectory::numTimeLevels)
      .def("__getitem__", &trajectoryLevel, py::arg("level"))
      .def_buffer([](TimeTrajectory& trajectory)
                  { return py::buffer_info(
                        trajectory.data(),
                        sizeof(Real),
                        py::format_descriptor<Real>::format(),
                        2,
                        {trajectory.numTimeLevels(), trajectory.numStates()},
                        {static_cast<py::ssize_t>(trajectory.numStates()
                                                  * sizeof(Real)),
                         static_cast<py::ssize_t>(sizeof(Real))}); });

  py::class_<PythonHostTimeIntegrator>(module, "TimeIntegrator")
      .def(py::init([](const TimeResidual&  problem,
                       HostCsrLinearSolver& solver)
                    { return std::make_unique<PythonHostTimeIntegrator>(
                          problem, solver); }),
           py::arg("problem"),
           py::arg("linear_solver"),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>())
      .def_property_readonly(
          "num_steps",
          [](const PythonHostTimeIntegrator& owner)
          { return owner.get().numSteps(); })
      .def_property_readonly(
          "num_states",
          [](const PythonHostTimeIntegrator& owner)
          { return owner.get().numStates(); })
      .def_property_readonly(
          "num_param",
          [](const PythonHostTimeIntegrator& owner)
          { return owner.get().numParams(); })
      .def(
          "solve",
          [](PythonHostTimeIntegrator& owner,
             const RealArray&          parameters,
             const py::object&         progress)
          {
            auto& integrator = owner.get();
            if (!progress.is_none() && !PyCallable_Check(progress.ptr()))
            {
              throw py::type_error("progress must be callable");
            }
            HostVector     vals = vectorFromArray(parameters, "parameters");
            TimeTrajectory trajectory;
            if (progress.is_none())
            {
              py::gil_scoped_release release;
              integrator.solve(vals.view(), trajectory);
            }
            else
            {
              PythonTimeObserver           observer(progress);
              HostTimeIntegrator::Observer callback =
                  [&observer](const TimeStepStateContext& context)
              {
                return observer(context);
              };
              py::gil_scoped_release release;
              integrator.solve(vals.view(), trajectory, callback);
            }
            return trajectory;
          },
          py::arg("param"),
          py::arg("progress") = py::none())
      .def(
          "set_initial_state",
          [](PythonHostTimeIntegrator& owner, const RealArray& state)
          {
            owner.get().setInitialState(
                vectorFromArray(state, "initial_state"));
          },
          py::arg("initial_state"))
      .def("clear_initial_state",
           [](PythonHostTimeIntegrator& owner)
           { owner.get().clearInitialState(); })
      .def("reset_timing",
           [](PythonHostTimeIntegrator& owner)
           { owner.get().resetStats(); })
      .def_property_readonly(
          "assembly_seconds",
          [](const PythonHostTimeIntegrator& owner)
          { return owner.get().lastStats().assm_sec; })
      .def_property_readonly(
          "solve_seconds",
          [](const PythonHostTimeIntegrator& owner)
          { return owner.get().lastStats().lin_solve_sec; })
      .def_property_readonly(
          "assembly_calls",
          [](const PythonHostTimeIntegrator& owner)
          { return owner.get().lastStats().assm_calls; })
      .def_property_readonly(
          "solve_calls",
          [](const PythonHostTimeIntegrator& owner)
          { return owner.get().lastStats().lin_solve_calls; });

  py::class_<HostInitialStateMap>(module, "_InitialStateMap")
      .def_property_readonly(
          "num_states", &HostInitialStateMap::numStates)
      .def_property_readonly(
          "num_param", &HostInitialStateMap::numParams)
      .def_property_readonly(
          "num_modes", &HostInitialStateMap::numModes)
      .def(
          "evaluate",
          [](const HostInitialStateMap& map,
             const RealArray&           param)
          {
            const HostVector prm = vectorFromArray(param, "param");
            HostVector       out(map.numStates());
            femx::fem::initialState(map, prm.view(), out.view());
            return vectorArray(out);
          },
          py::arg("param"));
}
