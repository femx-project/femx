#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "Bindings.hpp"
#include "InitialStateParameterMap.hpp"
#include "PETScInit.hpp"
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/DenseLinearSolver.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>
#ifdef FEMX_HAS_PETSC
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>
#include <femx/linalg/petsc/PETScVector.hpp>
#include <femx/runtime/PETScRuntime.hpp>
#endif
#ifdef FEMX_HAS_RESOLVE
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#endif
#include <femx/state/EnsembleBasis.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>
#include <femx/state/TimeResidual.hpp>
#include <femx/state/TimeStateMonitor.hpp>
#include <femx/state/TimeTrajectory.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{

using femx::DenseMatrix;
using femx::HostVector;
using femx::Index;
using femx::Real;
using femx::assembly::HostAssemblyMap;
using femx::linalg::DenseLinearSolver;
using femx::linalg::LinearOperator;
using femx::linalg::LinearSolver;
using femx::linalg::MapCsrMatrix;
using femx::linalg::MatrixOperator;
#ifdef FEMX_HAS_PETSC
using femx::linalg::KspLinearSolver;
using femx::linalg::PETScOperator;
#endif
#ifdef FEMX_HAS_RESOLVE
using femx::linalg::ReSolveLinearSolver;
using femx::linalg::ReSolveOptions;
#endif
using femx::state::EnsembleBasis;
using femx::state::TimeContext;
using femx::state::TimeDims;
using femx::state::TimeHistoryView;
using femx::state::TimeIntegrator;
using femx::state::TimeLinearIntegrator;
using femx::state::TimeLinearization;
using femx::state::TimeResidual;
using femx::state::TimeStateMonitor;
using femx::state::TimeStepStateContext;
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

class ScopedTimeStateMonitor
{
public:
  ScopedTimeStateMonitor(TimeIntegrator&   integrator,
                         TimeStateMonitor& monitor)
    : integrator_(integrator)
  {
    integrator_.setMonitor(&monitor);
  }

  ScopedTimeStateMonitor(const ScopedTimeStateMonitor&)            = delete;
  ScopedTimeStateMonitor& operator=(const ScopedTimeStateMonitor&) = delete;

  ~ScopedTimeStateMonitor()
  {
    integrator_.clearMonitor();
  }

private:
  TimeIntegrator& integrator_;
};

class PythonTimeStateMonitor final : public TimeStateMonitor
{
public:
  explicit PythonTimeStateMonitor(py::object progress)
    : progress_(std::move(progress))
  {
  }

  void observe(Index, const HostVector&) override
  {
    py::gil_scoped_acquire acquire;
    checkSignals();
  }

  bool observeStep(const TimeStepStateContext& ctx) override
  {
    py::gil_scoped_acquire acquire;
    checkSignals();
    if (progress_.is_none())
    {
      return false;
    }

    py::dict event;
    event["type"]                 = "solve";
    event["phase"]                = "forward";
    event["step"]                 = ctx.level;
    event["total"]                = ctx.total_steps;
    event["assembly_seconds"]     = ctx.assembly_sec;
    event["linear_solve_seconds"] = ctx.solve_sec;
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

class AffineInitialStateIntegrator final : public TimeIntegrator
{
public:
  AffineInitialStateIntegrator(TimeLinearIntegrator&     integrator,
                               InitialStateParameterMap& map)
    : integrator_(integrator),
      map_(map)
  {
    if (map_.numStates() != integrator_.numStates()
        || map_.numParams() != integrator_.numParams())
    {
      throw std::runtime_error(
          "initial-state map must match the time integrator");
    }
  }

  Index numSteps() const override
  {
    return integrator_.numSteps();
  }

  Index numStates() const override
  {
    return integrator_.numStates();
  }

  Index numParams() const override
  {
    return integrator_.numParams();
  }

  void solve(const HostVector& param,
             TimeTrajectory&   trajectory) override
  {
    if (param.size() != numParams())
    {
      throw std::runtime_error(
          "initial-state integrator parameter size mismatch");
    }

    HostVector initial;
    map_.state(param, initial);
    integrator_.setInitialState(initial);
    MonitorScope           monitor(*this);
    ForwardMonitor         forward_monitor(*this);
    ScopedTimeStateMonitor monitor_scope(integrator_, forward_monitor);
    integrator_.solve(param, trajectory);
  }

private:
  class ForwardMonitor final : public TimeStateMonitor
  {
  public:
    explicit ForwardMonitor(AffineInitialStateIntegrator& integrator)
      : integrator_(integrator)
    {
    }

    void observe(Index level, const HostVector& state) override
    {
      integrator_.observeState(level, state);
    }

    bool observeStep(const TimeStepStateContext& ctx) override
    {
      return integrator_.observeStep(ctx);
    }

  private:
    AffineInitialStateIntegrator& integrator_;
  };

  TimeLinearIntegrator&     integrator_;
  InitialStateParameterMap& map_;
};

py::array_t<Real> denseMatrixArray(const DenseMatrix& vals)
{
  py::array_t<Real> out({vals.numRows(), vals.numCols()});
  auto              data = out.mutable_unchecked<2>();
  for (Index row = 0; row < vals.numRows(); ++row)
  {
    for (Index col = 0; col < vals.numCols(); ++col)
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
  for (Index row = 0; row < vals.numRows(); ++row)
  {
    for (Index col = 0; col < vals.numCols(); ++col)
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
  if (perturb_vals.numRows() != mean_vals.size()
      || perturb_vals.numCols() <= 0)
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

py::array_t<Real> historyArray(const TimeHistoryView& history)
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

py::dict ctxData(const TimeContext& ctx)
{
  if (ctx.nxt == nullptr || ctx.prm == nullptr)
  {
    throw std::runtime_error("TimeContext contains null state or parameter data");
  }

  py::dict out;
  out["step"]       = ctx.step;
  out["next_state"] = vectorArray(*ctx.nxt);
  out["parameters"] = vectorArray(*ctx.prm);
  out["history"]    = historyArray(ctx.hist);
  return out;
}

Index variableSize(const TimeDims& dims, VariableBlock wrt)
{
  return wrt.isParam() ? dims.num_param : dims.num_states;
}

class PyTimeResidual : public TimeResidual
{
public:
  using TimeResidual::TimeResidual;

  TimeDims dims() const override
  {
    PYBIND11_OVERRIDE_PURE(TimeDims, TimeResidual, dims);
  }

  void res(const TimeContext& ctx,
           HostVector&        out) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override = py::get_override(this, "residual");
    if (!override)
    {
      throw std::runtime_error("TimeResidual.residual() is not implemented");
    }
    copyArray(override(ctxData(ctx)), out, "residual result");
  }

  void applyJac(const TimeContext& ctx,
                VariableBlock      wrt,
                const HostVector&  dir,
                HostVector&        out) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override = py::get_override(this, "apply_jacobian");
    if (!override)
    {
      throw std::runtime_error(
          "TimeResidual.apply_jacobian() is not implemented");
    }
    copyArray(override(ctxData(ctx), wrt, vectorArray(dir)),
              out,
              "Jacobian result");
  }

  void applyJacT(const TimeContext& ctx,
                 VariableBlock      wrt,
                 const HostVector&  adj,
                 HostVector&        out) const override
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

  bool assembleJac(const TimeContext& ctx,
                   VariableBlock      wrt,
                   MatrixOperator&    out) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override =
        py::get_override(this, "assemble_jacobian");
    if (!override)
    {
      return TimeResidual::assembleJac(ctx, wrt, out);
    }

    const py::object value = override(ctxData(ctx), wrt);
    if (value.is_none())
    {
      return false;
    }

    const RealArray mat = RealArray::ensure(value);
    if (!mat || mat.ndim() != 2)
    {
      throw std::runtime_error(
          "TimeResidual.assemble_jacobian() must return a two-dimensional array or None");
    }

    const TimeDims dims = this->dims();
    const Index    rows = static_cast<Index>(mat.shape(0));
    const Index    cols = static_cast<Index>(mat.shape(1));
    if (rows != dims.num_res || cols != variableSize(dims, wrt))
    {
      throw std::runtime_error(
          "TimeResidual.assemble_jacobian() returned an array with invalid shape");
    }

    if (out.numRows() != rows || out.numCols() != cols)
    {
      out.resize(rows, cols);
    }
    out.setZero();
    const auto data = mat.unchecked<2>();
    for (Index i = 0; i < rows; ++i)
    {
      for (Index j = 0; j < cols; ++j)
      {
        out.set(i, j, data(i, j));
      }
    }
    return true;
  }

  void prepareLinearSolve(const TimeContext& ctx,
                          VariableBlock      wrt,
                          MatrixOperator&    jac,
                          HostVector&        rhs) const override
  {
    py::gil_scoped_acquire gil;
    const py::function     override =
        py::get_override(this, "prepare_linear_solve");
    if (!override)
    {
      TimeResidual::prepareLinearSolve(ctx, wrt, jac, rhs);
      return;
    }

    py::array_t<Real> rhs_array = vectorArray(rhs);
    const py::object  result    = override(
        ctxData(ctx),
        wrt,
        py::cast(&jac, py::return_value_policy::reference),
        rhs_array);
    if (result.is_none())
    {
      copyArray(rhs_array, rhs, "prepared right-hand side");
    }
    else
    {
      copyArray(result, rhs, "prepared right-hand side");
    }
  }
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

py::array_t<Real> denseVals(const DenseMatrix& mat)
{
  py::array_t<Real> out({mat.numRows(), mat.numCols()});
  auto              vals = out.mutable_unchecked<2>();
  for (Index i = 0; i < mat.numRows(); ++i)
  {
    for (Index j = 0; j < mat.numCols(); ++j)
    {
      vals(i, j) = mat(i, j);
    }
  }
  return out;
}

#ifdef FEMX_HAS_PETSC
std::unique_ptr<PETScOperator> petscOperator(const HostAssemblyMap& map)
{
  femx::python::initializePETSc();

  auto                      mat = std::make_unique<PETScOperator>(PETSC_COMM_WORLD);
  femx::linalg::PETScVector lyt(PETSC_COMM_WORLD);
  lyt.resize(map.numRes());
  mat->resize(map.graph(), lyt);
  return mat;
}

std::unique_ptr<KspLinearSolver> kspLinearSolver()
{
  femx::python::initializePETSc();
  return std::make_unique<KspLinearSolver>(PETSC_COMM_WORLD);
}
#endif

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

  py::class_<LinearOperator>(module, "LinearOperator")
      .def_property_readonly("num_rows", &LinearOperator::numRows)
      .def_property_readonly("num_cols", &LinearOperator::numCols);

  py::class_<MatrixOperator, LinearOperator>(module, "MatrixOperator")
      .def("resize", &MatrixOperator::resize)
      .def("set_zero", &MatrixOperator::setZero)
      .def("set", &MatrixOperator::set)
      .def("add", &MatrixOperator::add)
      .def("finalize", &MatrixOperator::finalize);

  py::class_<HostAssemblyMap>(module, "_AssemblyMap");

  py::class_<MapCsrMatrix, MatrixOperator>(module,
                                           "_MapCsrMatrix")
      .def(py::init<const HostAssemblyMap&>(), py::arg("map"));

  py::class_<DenseMatrix, MatrixOperator>(module,
                                          "DenseMatrix")
      .def(py::init<>())
      .def(py::init([](Index rows, Index cols)
                    {
                      auto mat = std::make_unique<DenseMatrix>();
                      mat->resize(rows, cols);
                      return mat; }),
           py::arg("rows"),
           py::arg("cols"))
      .def_property_readonly("values", &denseVals);

  py::class_<LinearSolver>(module, "LinearSolver");
  py::class_<DenseLinearSolver, LinearSolver>(module, "DenseLinearSolver")
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
      .def_readwrite("preconditioner_side",
                     &ReSolveOptions::preconditioner_side)
      .def_readwrite("max_its", &ReSolveOptions::max_its)
      .def_readwrite("restart", &ReSolveOptions::restart)
      .def_readwrite("rtol", &ReSolveOptions::rtol)
      .def_readwrite("flexible", &ReSolveOptions::flexible);

  py::class_<ReSolveLinearSolver, LinearSolver>(module,
                                                "_ReSolveLinearSolver")
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

  py::class_<PETScOperator, MatrixOperator>(module,
                                            "_PETScOperator")
      .def(py::init(&petscOperator), py::arg("map"));

  py::class_<KspLinearSolver, LinearSolver>(module, "_KspLinearSolver")
      .def(py::init(&kspLinearSolver));
#endif

  py::class_<TimeDims>(module, "TimeDims")
      .def(py::init<>())
      .def_readwrite("num_steps", &TimeDims::num_steps)
      .def_readwrite("num_states", &TimeDims::num_states)
      .def_readwrite("num_param", &TimeDims::num_param)
      .def_readwrite("num_res", &TimeDims::num_res)
      .def_readwrite("num_history_states", &TimeDims::num_history_states);

  py::class_<VariableBlock>(module, "VariableBlock")
      .def_static("history", &VariableBlock::hist, py::arg("lag"))
      .def_property_readonly("is_history_state",
                             &VariableBlock::isHistoryState)
      .def_property_readonly("is_next_state", &VariableBlock::isNextState)
      .def_property_readonly("is_parameter", &VariableBlock::isParam)
      .def_property_readonly("history_lag",
                             &VariableBlock::historyLag);

  py::class_<TimeResidual, PyTimeResidual>(module, "TimeResidual")
      .def(py::init<>())
      .def("dims", &TimeResidual::dims);

  py::class_<TimeLinearization>(module, "TimeLinearization")
      .def(py::init<>());

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

  py::class_<TimeIntegrator>(module, "TimeIntegrator")
      .def_property_readonly("num_steps", &TimeIntegrator::numSteps)
      .def_property_readonly("num_states", &TimeIntegrator::numStates)
      .def_property_readonly("num_param", &TimeIntegrator::numParams)
      .def(
          "solve",
          [](TimeIntegrator&   integrator,
             const RealArray&  parameters,
             const py::object& progress)
          {
            if (!progress.is_none() && !PyCallable_Check(progress.ptr()))
            {
              throw py::type_error("progress must be callable");
            }
            HostVector             vals = vectorFromArray(parameters, "parameters");
            TimeTrajectory         trajectory;
            PythonTimeStateMonitor monitor(progress);
            ScopedTimeStateMonitor monitor_scope(integrator, monitor);
            {
              py::gil_scoped_release release;
              integrator.solve(vals, trajectory);
            }
            return trajectory;
          },
          py::arg("param"),
          py::arg("progress") = py::none());

  py::class_<TimeLinearIntegrator, TimeIntegrator>(
      module, "TimeLinearIntegrator")
      .def(py::init<const TimeResidual&, MatrixOperator&, LinearSolver&>(),
           py::arg("problem"),
           py::arg("matrix"),
           py::arg("linear_solver"),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>(),
           py::keep_alive<1, 4>())
      .def(
          "set_initial_state",
          [](TimeLinearIntegrator& integrator, const RealArray& state)
          {
            integrator.setInitialState(
                vectorFromArray(state, "initial_state"));
          },
          py::arg("initial_state"))
      .def("clear_initial_state",
           &TimeLinearIntegrator::clearInitialState)
      .def("reset_timing", &TimeLinearIntegrator::resetTiming)
      .def_property_readonly("assembly_seconds",
                             &TimeLinearIntegrator::assemblySeconds)
      .def_property_readonly("solve_seconds",
                             &TimeLinearIntegrator::solveSeconds)
      .def_property_readonly("last_assembly_seconds",
                             &TimeLinearIntegrator::lastAssemblySeconds)
      .def_property_readonly("last_solve_seconds",
                             &TimeLinearIntegrator::lastSolveSeconds)
      .def_property_readonly("assembly_calls",
                             &TimeLinearIntegrator::assemblyCalls)
      .def_property_readonly("solve_calls",
                             &TimeLinearIntegrator::solveCalls);

  py::class_<InitialStateParameterMap>(module, "_InitialStateParameterMap")
      .def_property_readonly(
          "num_states", &InitialStateParameterMap::numStates)
      .def_property_readonly(
          "num_param", &InitialStateParameterMap::numParams)
      .def_property_readonly(
          "num_modes", &InitialStateParameterMap::numModes)
      .def(
          "evaluate",
          [](const InitialStateParameterMap& map,
             const RealArray&                param)
          {
            HostVector out;
            map.state(vectorFromArray(param, "param"), out);
            return vectorArray(out);
          },
          py::arg("param"));

  py::class_<AffineInitialStateIntegrator, TimeIntegrator>(
      module, "_AffineInitialStateIntegrator")
      .def(py::init<TimeLinearIntegrator&, InitialStateParameterMap&>(),
           py::arg("integrator"),
           py::arg("map"),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>());
}
