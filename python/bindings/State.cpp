#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "Bindings.hpp"
#include "NumpyConversions.hpp"
#include "PETScInit.hpp"
#include <femx/common/Types.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/SystemMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/HostLinearSystem.hpp>
#ifdef FEMX_HAS_PETSC
#include <femx/linalg/petsc/PETScLinearSystem.hpp>
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
using femx::HostVector;
using femx::HostVectorView;
using femx::Index;
using femx::MemorySpace;
using femx::Real;
using femx::fem::HostInitialStateMap;
#ifdef FEMX_HAS_RESOLVE
using femx::linalg::ReSolveLinearSolver;
using femx::linalg::ReSolveOptions;
#endif
using femx::python::bindings::denseMatrixArray;
using femx::python::bindings::denseMatrixFromArray;
using femx::python::bindings::FiniteCheck;
using femx::python::bindings::RealArray;
using femx::python::bindings::requireFinite;
using femx::python::bindings::vectorArray;
using femx::python::bindings::vectorFromArray;
using femx::runtime::SolverType;
using femx::state::EnsembleBasis;
using femx::state::HostTimeContext;
using femx::state::HostTimeHistoryView;
using femx::state::HostTimeIntegrator;
using femx::state::TimeDims;
using femx::state::TimeStepStateContext;
using TimeResidual = femx::state::HostTimeResidual;
using femx::state::TimeTrajectory;
using femx::state::VariableBlock;

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

EnsembleBasis ensembleBasisFromArrays(const RealArray& mean,
                                      const RealArray& perturbations)
{
  HostVector<Real> mean_vals =
      vectorFromArray(mean, "mean", FiniteCheck::Skip);
  DenseMatrix perturb_vals = denseMatrixFromArray(
      perturbations, "perturbations", FiniteCheck::Skip);
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
  requireFinite(mean_vals, "mean");
  requireFinite(perturb_vals, "perturbations");
  return EnsembleBasis(
      std::move(mean_vals), std::move(perturb_vals));
}

void copyArray(const py::handle& value,
               HostVector<Real>& out,
               const char*       name)
{
  const RealArray vals = RealArray::ensure(value);
  if (!vals)
  {
    throw std::runtime_error(std::string(name) + " must be a real NumPy array");
  }
  out = vectorFromArray(vals, name, FiniteCheck::Skip);
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

  const femx::HostCsrPattern& hostPattern() const override
  {
    updateGraph();
    return pattern_;
  }

  void initialState(HostVectorView<const Real>                      prm,
                    HostVector<Real>&                               out,
                    femx::linalg::Context<femx::MemorySpace::Host>& ctx)
      const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override = py::get_override(this, "initial_state");
    if (!override)
    {
      ctx.vectorHandler().resizeOrZero(out, dims().num_states);
      return;
    }
    copyArray(override(vectorArray(prm)), out, "initial state");
  }

  void addInitialStateJacT(
      HostVectorView<const Real>                      state_grad,
      HostVectorView<Real>                            out,
      femx::linalg::Context<femx::MemorySpace::Host>& ctx) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override =
        py::get_override(this, "add_initial_state_jac_transpose");
    if (!override)
    {
      TimeResidual::addInitialStateJacT(state_grad, out, ctx);
      return;
    }
    HostVector<Real> grad;
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

  void applyJacT(const HostTimeContext&     ctx,
                 VariableBlock              wrt,
                 HostVectorView<const Real> adj,
                 HostVector<Real>&          out,
                 femx::linalg::Context<femx::MemorySpace::Host>&)
      const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override =
        py::get_override(this, "apply_jac_transpose");
    if (!override)
    {
      throw std::runtime_error(
          "TimeResidual.apply_jac_transpose() is not implemented");
    }
    copyArray(override(ctxData(ctx), wrt, vectorArray(adj)),
              out,
              "transpose Jacobian result");
  }

  void assembleNext(const HostTimeContext&                               ctx,
                    HostVector<Real>&                                    res_out,
                    femx::linalg::SystemMatrix<femx::MemorySpace::Host>& jac,
                    femx::linalg::Context<femx::MemorySpace::Host>&)
      const override
  {
    evaluateResidual(ctx, res_out);
    updateGraph();

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

    DenseMatrix       values(rows, cols);
    HostVector<Index> jac_rows(rows);
    HostVector<Index> jac_columns(cols);
    HostVector<Index> csr_entries(rows * cols);
    const auto        data = mat.unchecked<2>();
    for (Index row = 0; row < rows; ++row)
    {
      jac_rows[row] = row;
      for (Index col = 0; col < cols; ++col)
      {
        values(row, col)              = data(row, col);
        csr_entries[row * cols + col] = row * cols + col;
      }
    }
    for (Index col = 0; col < cols; ++col)
    {
      jac_columns[col] = col;
    }
    jac.addElement(
        {jac_rows.view(),
         jac_columns.view(),
         csr_entries.view(),
         values.view()});
  }

  void setup(const HostTimeContext& ctx,
             femx::linalg::SystemMatrix<femx::MemorySpace::Host>&,
             HostVector<Real>& rhs,
             femx::linalg::Context<femx::MemorySpace::Host>&)
      const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override = py::get_override(this, "setup");
    if (!override)
    {
      return;
    }

    py::array_t<Real> rhs_array = vectorArray(rhs);
    const py::object  result    = override(ctxData(ctx), rhs_array);
    if (result.is_none())
    {
      copyArray(rhs_array, rhs, "prepared right-hand side");
    }
    else
    {
      copyArray(result, rhs, "prepared right-hand side");
    }
  }

private:
  void evaluateResidual(const HostTimeContext& ctx, HostVector<Real>& out) const
  {
    py::gil_scoped_acquire gil;
    const py::function     override = py::get_override(this, "residual");
    if (!override)
    {
      throw std::runtime_error("TimeResidual.residual() is not implemented");
    }
    copyArray(override(ctxData(ctx)), out, "residual result");
  }

  void updateGraph() const
  {
    const TimeDims dim = dims();
    if (pattern_.rows() == dim.num_res && pattern_.cols() == dim.num_states)
    {
      return;
    }
    femx::HostVector<Index> row_ptr(dim.num_res + 1);
    femx::HostVector<Index> col_ind(dim.num_res * dim.num_states);
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
    pattern_ = femx::HostCsrPattern(
        dim.num_res, dim.num_states, std::move(row_ptr), std::move(col_ind));
  }

  mutable femx::HostCsrPattern pattern_;
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

std::unique_ptr<femx::linalg::LinearSystem<femx::MemorySpace::Host>>
makePythonHostLinearSystem(SolverType        solver,
                           const py::object& options)
{
  if (solver == SolverType::Dense)
  {
    if (!options.is_none())
    {
      throw py::value_error(
          "Dense solver options are not supported");
    }
    return std::make_unique<femx::linalg::HostLinearSystem>();
  }

  if (solver == SolverType::ReSolve)
  {
#if defined(FEMX_HAS_RESOLVE)
    const ReSolveOptions opts =
        options.is_none() ? ReSolveOptions{}
                          : options.cast<ReSolveOptions>();
    auto native_solver =
        std::make_unique<ReSolveLinearSolver>(opts);
    return std::make_unique<femx::linalg::HostLinearSystem>(
        std::move(native_solver));
#else
    static_cast<void>(options);
    throw py::value_error(
        "ReSolve is unavailable in this femx build");
#endif
  }

  if (solver == SolverType::PETSc)
  {
    if (!options.is_none())
    {
      throw py::value_error(
          "PETSc solver options are not supported");
    }
#if defined(FEMX_HAS_PETSC)
    femx::python::initializePETSc();
    return std::make_unique<femx::linalg::PETScLinearSystem>(
        PETSC_COMM_WORLD);
#else
    throw py::value_error(
        "PETSc is unavailable in this femx build");
#endif
  }

  throw py::value_error("Unknown linear solver selection");
}

void bindState(py::module_& module)
{
  py::enum_<MemorySpace>(module, "MemorySpace")
      .value("HOST", MemorySpace::Host)
      .value("DEVICE", MemorySpace::Device);

  py::enum_<SolverType>(module, "SolverType")
      .value("DENSE", SolverType::Dense)
      .value("RESOLVE", SolverType::ReSolve)
      .value("PETSC", SolverType::PETSc);

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
            HostVector<Real> coeffs = vectorFromArray(
                coefficients, "coefficients", FiniteCheck::Require);
            HostVector<Real> out;
            basis.apply(coeffs, out);
            return vectorArray(out);
          },
          py::arg("coefficients"))
      .def(
          "apply_transpose",
          [](const EnsembleBasis& basis, const RealArray& vals)
          {
            HostVector<Real> phys =
                vectorFromArray(vals, "values", FiniteCheck::Require);
            HostVector<Real> out;
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
      .def(py::init<const TimeResidual&,
                    SolverType,
                    const py::object&>(),
           py::arg("problem"),
           py::arg("solver")  = SolverType::Dense,
           py::arg("options") = py::none(),
           py::keep_alive<1, 2>())
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
            HostVector<Real> vals = vectorFromArray(
                parameters, "parameters", FiniteCheck::Skip);
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
                vectorFromArray(
                    state, "initial_state", FiniteCheck::Skip));
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
            const HostVector<Real> prm =
                vectorFromArray(param, "param", FiniteCheck::Skip);
            HostVector<Real> out(map.numStates());
            femx::fem::initialState(map, prm.view(), out.view());
            return vectorArray(out);
          },
          py::arg("param"));
}
