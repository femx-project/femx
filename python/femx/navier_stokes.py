"""Navier-Stokes model construction backed by the femx C++ engine."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from . import _core
from .mesh import Mesh


_BACKENDS = {}

_RESOLVE_STRING_OPTIONS = (
    "factor",
    "refactor",
    "solve",
    "precond",
    "ir",
    "gram_schmidt",
    "sketching",
    "preconditioner_side",
)
_RESOLVE_INT_OPTIONS = ("max_its", "restart")
_RESOLVE_REAL_OPTIONS = ("rtol",)
_RESOLVE_BOOL_OPTIONS = ("flexible",)
_RESOLVE_OPTIONS = frozenset(
    _RESOLVE_STRING_OPTIONS
    + _RESOLVE_INT_OPTIONS
    + _RESOLVE_REAL_OPTIONS
    + _RESOLVE_BOOL_OPTIONS
)


def _factories(backend):
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    try:
        return _BACKENDS[backend]
    except KeyError as error:
        available = ", ".join(_BACKENDS)
        raise ValueError(
            f"unknown solver backend '{backend}'; available: {available}"
        ) from error


def solver_backends():
    """Return the solver backends available in this femx build."""

    return tuple(_BACKENDS)


def petsc_rank():
    """Return this process's rank in ``PETSC_COMM_WORLD``."""

    if not hasattr(_core, "_petsc_world_rank"):
        raise RuntimeError("femx was built without PETSc")
    return int(_core._petsc_world_rank())


def petsc_size():
    """Return the size of ``PETSC_COMM_WORLD``."""

    if not hasattr(_core, "_petsc_world_size"):
        raise RuntimeError("femx was built without PETSc")
    return int(_core._petsc_world_size())


def petsc_barrier():
    """Synchronize all processes in ``PETSC_COMM_WORLD``."""

    if not hasattr(_core, "_petsc_world_barrier"):
        raise RuntimeError("femx was built without PETSc")
    _core._petsc_world_barrier()


def petsc_finalize():
    """Finalize PETSc after all PETSc-backed objects have been released."""

    if not hasattr(_core, "_petsc_finalize"):
        raise RuntimeError("femx was built without PETSc")
    _core._petsc_finalize()


def _resolve_options(values):
    if not isinstance(values, Mapping):
        raise TypeError("options must be a mapping")

    unknown = set(values).difference(_RESOLVE_OPTIONS)
    if unknown:
        names = ", ".join(sorted((repr(name) for name in unknown)))
        raise ValueError(f"unknown ReSolve option(s): {names}")

    options = _core._ReSolveOptions()
    for name in _RESOLVE_STRING_OPTIONS:
        if name not in values:
            continue
        value = values[name]
        if not isinstance(value, str):
            raise TypeError(f"options['{name}'] must be a string")
        if not value:
            raise ValueError(f"options['{name}'] must not be empty")
        setattr(options, name, value)

    for name in _RESOLVE_INT_OPTIONS:
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, Integral
        ):
            raise TypeError(f"options['{name}'] must be an integer")
        if value <= 0:
            raise ValueError(f"options['{name}'] must be positive")
        setattr(options, name, int(value))

    for name in _RESOLVE_REAL_OPTIONS:
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"options['{name}'] must be a real number")
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"options['{name}'] must be finite and positive")
        setattr(options, name, value)

    for name in _RESOLVE_BOOL_OPTIONS:
        if name not in values:
            continue
        value = values[name]
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"options['{name}'] must be a boolean")
        setattr(options, name, bool(value))

    return options


def _backend_options(backend, values):
    if values is None:
        return None
    if backend != "resolve":
        raise ValueError(
            f"solver options are not supported for backend '{backend}'"
        )
    return _resolve_options(values)


def _bnds(bnds, size):
    if bnds is None:
        return None, None

    if hasattr(bnds, "lb") and hasattr(bnds, "ub"):
        try:
            lower = np.broadcast_to(
                np.asarray(bnds.lb, dtype=float), (size,)
            ).copy()
            upper = np.broadcast_to(
                np.asarray(bnds.ub, dtype=float), (size,)
            ).copy()
        except ValueError as error:
            raise ValueError(
                f"bnds must contain {size} lower and upper values"
            ) from error
    else:
        try:
            entries = tuple(bnds)
        except TypeError as error:
            raise TypeError("bnds must be a sequence of pairs") from error
        if len(entries) != size:
            raise ValueError(f"bnds must contain {size} pairs")

        lower = np.empty(size)
        upper = np.empty(size)
        for i, entry in enumerate(entries):
            try:
                lo, hi = entry
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "each bound must be a (lower, upper) pair"
                ) from error
            lower[i] = -np.inf if lo is None else float(lo)
            upper[i] = np.inf if hi is None else float(hi)

    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("bnds must not contain NaN")
    if np.any(lower > upper):
        raise ValueError("each lower bound must not exceed its upper bound")
    return np.ascontiguousarray(lower), np.ascontiguousarray(upper)


def _configure_assembly(prob, backend):
    if backend == "petsc":
        prob.model._impl._use_petsc_world_element_range()
    else:
        prob.model._impl._use_full_element_range()


def _dense_matrix(prob):
    _configure_assembly(prob, "dense")
    return _core.DenseMatrix()


def _dense_solver(_, options=None):
    del options
    return _core.DenseLinearSolver()


def _resolve_matrix(prob):
    _configure_assembly(prob, "resolve")
    return _core._MapCsrMatrix(prob.model._impl._map)


def _resolve_solver(_, options=None):
    if options is None:
        return _core._ReSolveLinearSolver()
    return _core._ReSolveLinearSolver(options)


def _petsc_matrix(prob):
    _configure_assembly(prob, "petsc")
    return _core._PETScOperator(prob.model._impl._map)


def _petsc_solver(_, options=None):
    del options
    return _core._KspLinearSolver()


@dataclass(frozen=True)
class DirichletBC:
    """A fixed Dirichlet condition on a named or tagged boundary."""

    boundary: object
    field: str
    value: object

    def __post_init__(self):
        if isinstance(self.boundary, bool) or not isinstance(
            self.boundary, (str, int)
        ):
            raise TypeError("boundary must be a physical name or tag")
        if isinstance(self.boundary, str) and not self.boundary:
            raise ValueError("boundary name must not be empty")
        if self.field not in ("velocity", "pressure"):
            raise ValueError("field must be 'velocity' or 'pressure'")


@dataclass(frozen=True)
class NormalVelocityControl:
    """Scalar nodal inlet control mapped onto a fixed normal direction."""

    boundary: object
    normal: object
    times: object = None
    periodic: bool = False
    kind = "normal"

    def __post_init__(self):
        if isinstance(self.boundary, bool) or not isinstance(
            self.boundary, (str, int)
        ):
            raise TypeError("boundary must be a physical name or tag")
        if isinstance(self.boundary, str) and not self.boundary:
            raise ValueError("boundary name must not be empty")

        normal = np.asarray(self.normal, dtype=float)
        if normal.ndim != 1 or normal.size == 0:
            raise ValueError("normal must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) == 0.0:
            raise ValueError("normal must be finite and nonzero")
        object.__setattr__(self, "normal", tuple(float(x) for x in normal))

        if self.times is not None:
            times = np.asarray(self.times, dtype=float)
            if times.ndim != 1 or times.size == 0:
                raise ValueError(
                    "times must be a nonempty one-dimensional array"
                )
            if (
                not np.all(np.isfinite(times))
                or np.any(times < 0.0)
                or np.any(np.diff(times) <= 0.0)
            ):
                raise ValueError(
                    "times must be finite, nonnegative, and increasing"
                )
            object.__setattr__(
                self, "times", tuple(float(x) for x in times)
            )
        if not isinstance(self.periodic, (bool, np.bool_)):
            raise TypeError("periodic must be a boolean")
        if self.periodic and self.times is None:
            raise ValueError("periodic control requires explicit times")


@dataclass(frozen=True)
class VelocityControl:
    """Vector-valued nodal velocity control on a named boundary."""

    boundary: object
    times: object = None
    periodic: bool = False
    kind = "vector"

    def __post_init__(self):
        if isinstance(self.boundary, bool) or not isinstance(
            self.boundary, (str, int)
        ):
            raise TypeError("boundary must be a physical name or tag")
        if isinstance(self.boundary, str) and not self.boundary:
            raise ValueError("boundary name must not be empty")
        if self.times is not None:
            times = np.asarray(self.times, dtype=float)
            if times.ndim != 1 or times.size == 0:
                raise ValueError("times must be a nonempty one-dimensional array")
            if (
                not np.all(np.isfinite(times))
                or np.any(times < 0.0)
                or np.any(np.diff(times) <= 0.0)
            ):
                raise ValueError(
                    "times must be finite, nonnegative, and increasing"
                )
            object.__setattr__(self, "times", tuple(float(x) for x in times))
        if not isinstance(self.periodic, (bool, np.bool_)):
            raise TypeError("periodic must be a boolean")
        if self.periodic and self.times is None:
            raise ValueError("periodic control requires explicit times")


@dataclass(frozen=True)
class InitialStateControl:
    """Affine low-dimensional parameterization of the initial state."""

    modes: object
    mean: object = None

    def __post_init__(self):
        modes = np.asarray(self.modes, dtype=float)
        if modes.ndim != 2 or modes.shape[1] == 0:
            raise ValueError(
                "modes must have shape (num_states, num_modes)"
            )
        if not np.all(np.isfinite(modes)):
            raise ValueError("modes must be finite")
        object.__setattr__(self, "modes", np.ascontiguousarray(modes))

        if self.mean is not None:
            mean = np.asarray(self.mean, dtype=float)
            if mean.ndim != 1 or not np.all(np.isfinite(mean)):
                raise ValueError("mean must be a finite state vector")
            object.__setattr__(self, "mean", np.ascontiguousarray(mean))

    @property
    def num_modes(self):
        return self.modes.shape[1]


@dataclass(frozen=True)
class TaoResult:
    """Result of a PETSc/TAO reduced optimization."""

    param: np.ndarray
    grad: np.ndarray
    obj: float
    num_iter: int
    num_fun: int
    num_grad: int
    converged: bool
    status: int
    msg: str


class NavierStokesModel:
    """Reusable Navier-Stokes finite-element discretization."""

    def __init__(self, mesh_file, num_steps, dt, *, rho=1.0, mu=1.0):
        fluid = _core.FluidParams()
        fluid.density = rho
        fluid.dynamic_viscosity = mu
        self._impl = _core.NavierStokesModel(
            str(mesh_file), num_steps, dt, fluid
        )

    @property
    def num_steps(self):
        return self._impl.num_steps

    @property
    def num_states(self):
        return self._impl.num_states

    @property
    def dt(self):
        return self._impl.dt

    @property
    def rho(self):
        return self._impl.fluid.density

    @property
    def mu(self):
        return self._impl.fluid.dynamic_viscosity

    @property
    def mesh(self):
        return Mesh(self._impl.mesh)

    @property
    def residual(self):
        return self._impl.residual

    @property
    def velocity_dofs(self):
        return self._impl.velocity_dofs

    def velocity_boundary_dofs(self, selector):
        return self._impl.velocity_boundary_dofs(selector)

    def write_xdmf(self, path, traj):
        """Write velocity and pressure time series to HDF5/XDMF."""

        if not isinstance(traj, _core.TimeTrajectory):
            raise TypeError("traj must be a TimeTrajectory")
        try:
            self._impl.write_xdmf(str(path), traj)
        except RuntimeError as error:
            raise ValueError(str(error)) from error


class NavierStokesProblem:
    """A Navier-Stokes model with boundary values and optional inlet control."""

    def __init__(self, model):
        if not isinstance(model, NavierStokesModel):
            raise TypeError("model must be a NavierStokesModel")
        self._model = model
        self._bcs = []
        self._ctr = None
        self._init_ctr = None
        self._compiled = None
        self._solvers = {}
        self._fixed_dofs = np.empty(0, dtype=np.int64)
        self._fixed_values = np.empty((model.num_steps, 0), dtype=float)
        self._initial_state_value = None
        self._initial_state = np.zeros(model.num_states, dtype=float)
        self._initial_modes = np.empty((model.num_states, 0), dtype=float)
        self._initial_map = None

    @property
    def model(self):
        return self._model

    @property
    def bcs(self):
        return tuple(self._bcs)

    @property
    def ctr(self):
        return self._ctr

    @property
    def init_ctr(self):
        return self._init_ctr

    def add_bc(self, bc):
        if not isinstance(bc, DirichletBC):
            raise TypeError("bc must be a DirichletBC")
        self._bcs.append(bc)
        self._compiled = None
        self._solvers.clear()

    def add_ctr(self, ctr):
        if not isinstance(ctr, (NormalVelocityControl, VelocityControl)):
            raise TypeError("ctr must be a NormalVelocityControl or VelocityControl")
        if self._ctr is not None:
            raise ValueError("only one velocity control is supported")
        self._ctr = ctr
        self._compiled = None
        self._solvers.clear()

    def add_init_ctr(self, ctr):
        if not isinstance(ctr, InitialStateControl):
            raise TypeError("ctr must be an InitialStateControl")
        if self._init_ctr is not None:
            raise ValueError("only one initial-state control is supported")
        if ctr.modes.shape[0] != self._model.num_states:
            raise ValueError("initial-state modes must match model.num_states")
        if ctr.mean is not None and ctr.mean.size != self._model.num_states:
            raise ValueError("initial-state mean must match model.num_states")
        self._init_ctr = ctr
        self._compiled = None
        self._solvers.clear()

    def set_initial_state(self, state):
        """Set a fixed initial state without adding optimization parameters."""

        state = np.asarray(state, dtype=float)
        if state.shape != (self._model.num_states,):
            raise ValueError("state must match model.num_states")
        if not np.all(np.isfinite(state)):
            raise ValueError("state must be finite")
        self._initial_state_value = np.ascontiguousarray(state).copy()
        self._compiled = None
        self._solvers.clear()

    def build(self):
        self._compiled = None
        self._solvers.clear()
        if self._init_ctr is not None and self._ctr is None:
            raise ValueError(
                "initial-state control currently requires an inlet control"
            )
        try:
            if self._ctr is None:
                self._compiled = _core._FixedDirichletProblem(
                    self._model._impl,
                    self._bcs,
                )
            else:
                self._compiled = _core._ControlledDirichletProblem(
                    self._model._impl,
                    self._bcs,
                    self._ctr,
                    (
                        0
                        if self._init_ctr is None
                        else self._init_ctr.num_modes
                    ),
                )
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        self._fixed_dofs = np.asarray(self._compiled.fixed_dofs)
        self._fixed_values = np.asarray(self._compiled.fixed_values)
        self._initial_state = np.asarray(self._compiled.initial_state)
        if self._initial_state_value is not None:
            if not np.allclose(
                self._initial_state_value[self._fixed_dofs],
                self._initial_state[self._fixed_dofs],
            ):
                raise ValueError(
                    "initial state must preserve fixed boundary values"
                )
            self._initial_state = self._initial_state_value.copy()
        self._initial_modes = np.empty(
            (self._model.num_states, 0), dtype=float
        )
        self._initial_map = None
        if self._init_ctr is not None:
            mean = (
                self._initial_state
                if self._init_ctr.mean is None
                else self._init_ctr.mean
            )
            modes = self._init_ctr.modes
            if not np.allclose(
                mean[self._fixed_dofs],
                self._initial_state[self._fixed_dofs],
            ):
                raise ValueError(
                    "initial-state mean must preserve fixed boundary values"
                )
            if np.any(modes[self._fixed_dofs] != 0.0):
                raise ValueError(
                    "initial-state modes must vanish on fixed boundary dofs"
                )
            if np.any(modes[self.ctr_state_dofs] != 0.0):
                raise ValueError(
                    "initial-state modes must vanish on control boundary dofs"
                )
            self._initial_state = np.ascontiguousarray(mean).copy()
            self._initial_modes = np.ascontiguousarray(modes).copy()
        if self._init_ctr is not None:
            try:
                self._initial_map = self._compiled.make_initial_state_map(
                    self._initial_state,
                    self._initial_modes,
                )
            except RuntimeError as error:
                raise ValueError(str(error)) from error
        return self

    def _ensure_built(self):
        if self._compiled is None:
            self.build()

    @property
    def residual(self):
        self._ensure_built()
        return self._compiled.residual

    @property
    def initial_state(self):
        self._ensure_built()
        return self._initial_state.copy()

    @property
    def param(self):
        self._ensure_built()
        return np.zeros(self.residual.dims().num_param, dtype=float)

    @property
    def num_param(self):
        self._ensure_built()
        return self.residual.dims().num_param

    @property
    def num_init_param(self):
        self._ensure_built()
        return self._initial_modes.shape[1]

    @property
    def ctr_param_offset(self):
        return self.num_init_param

    @property
    def num_ctr_param(self):
        return self.num_param - self.num_init_param

    @property
    def num_ctr_params_per_level(self):
        self._ensure_built()
        if self._ctr is None:
            return 0
        return self._compiled.num_ctr_parameters

    @property
    def num_ctr_levels(self):
        self._ensure_built()
        if self._ctr is None:
            return 0
        return self._compiled.num_ctr_levels

    @property
    def param_shape(self):
        return self.num_ctr_levels, self.num_ctr_params_per_level

    @property
    def initial_state_modes(self):
        self._ensure_built()
        return self._initial_modes.copy()

    def initial_state_from_param(self, param):
        param = self._check_param(param)
        if self._initial_map is None:
            return self._initial_state.copy()
        return np.asarray(self._initial_map.evaluate(param))

    @property
    def ctr_state_dofs(self):
        self._ensure_built()
        if self._ctr is None:
            return np.empty(0, dtype=np.int64)
        return np.asarray(self._compiled.ctr_state_dofs).copy()

    @property
    def ctr_mesh_node_ids(self):
        self._ensure_built()
        if self._ctr is None:
            return np.empty(0, dtype=np.int64)
        return np.asarray(self._compiled.ctr_mesh_node_ids).copy()

    @property
    def fixed_dofs(self):
        self._ensure_built()
        return self._fixed_dofs.copy()

    @property
    def fixed_values(self):
        self._ensure_built()
        return self._fixed_values.copy()

    def create_solver(self, backend="dense", options=None):
        """Create a forward solver, with optional ReSolve settings."""

        return NavierStokesSolver(self, backend, options=options)

    def create_reduced(self, obj, backend="dense"):
        """Create a reduced functional for the selected backend."""

        return NavierStokesReducedFunctional(self, obj, backend)

    def _check_param(self, param):
        values = self.param if param is None else param
        values = np.ascontiguousarray(values, dtype=float).reshape(-1)
        if values.size != self.num_param:
            raise ValueError(f"param must contain {self.num_param} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("param must be finite")
        return values

    def solve(self, param=None, backend="dense"):
        if backend not in self._solvers:
            self._solvers[backend] = self.create_solver(backend)
        return self._solvers[backend].solve(param)


class NavierStokesSolver:
    """Reusable time integrator for one compiled problem and backend."""

    def __init__(self, prob, backend, options=None):
        if not isinstance(prob, NavierStokesProblem):
            raise TypeError("prob must be a NavierStokesProblem")
        matrix, solver = _factories(backend)
        options = _backend_options(backend, options)
        prob._ensure_built()

        self._prob = prob
        self._compiled = prob._compiled
        self._backend = backend
        self._matrix = matrix(prob)
        self._linear_solver = solver(prob, options)
        self._linear_integrator = _core.TimeLinearIntegrator(
            prob.residual,
            self._matrix,
            self._linear_solver,
        )
        self._linear_integrator.set_initial_state(prob._initial_state)
        if prob._initial_map is not None:
            self._integrator = _core._AffineInitialStateIntegrator(
                self._linear_integrator,
                prob._initial_map,
            )
        else:
            self._integrator = self._linear_integrator
        self._num_solves = 0

    @property
    def prob(self):
        return self._prob

    @property
    def backend(self):
        return self._backend

    @property
    def num_solves(self):
        return self._num_solves

    @property
    def assembly_calls(self):
        return self._linear_integrator.assembly_calls

    @property
    def linear_solve_calls(self):
        return self._linear_integrator.solve_calls

    def solve(self, param=None, progress=None):
        if self._prob._compiled is not self._compiled:
            raise RuntimeError(
                "problem configuration changed; create a new solver"
            )
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")

        _configure_assembly(self._prob, self._backend)
        param = self._prob._check_param(param)
        traj = self._integrator.solve(param, progress=progress)
        self._num_solves += 1
        return traj


class NavierStokesReducedFunctional:
    """Reusable forward and adjoint evaluation for one solver backend."""

    def __init__(self, prob, obj, backend):
        if not isinstance(prob, NavierStokesProblem):
            raise TypeError("prob must be a NavierStokesProblem")
        if not isinstance(obj, _core.TimeObjective):
            raise TypeError("obj must be a TimeObjective")
        matrix, solver = _factories(backend)
        prob._ensure_built()
        if (
            obj.num_steps != prob.model.num_steps
            or obj.num_states != prob.model.num_states
            or obj.num_param != prob.num_param
        ):
            raise ValueError("obj dimensions must match the problem")

        self._prob = prob
        self._compiled = prob._compiled
        self._obj = obj
        self._backend = backend
        self._forward_solver = prob.create_solver(backend)
        self._linearization = _core.TimeLinearization()
        self._next_state_matrix = matrix(prob)
        self._history_matrix = matrix(prob)
        self._adjoint_solver = solver(prob)
        self._impl = _core.TimeReducedFunctional(
            self._forward_solver._integrator,
            prob.residual,
            self._linearization,
            self._next_state_matrix,
            self._history_matrix,
            self._adjoint_solver,
            obj,
        )
        self._init_grad = None
        if prob._initial_map is not None:
            self._init_grad = _core._AffineInitialStateGradientMap(
                prob._initial_map,
            )
            self._impl.set_initial_state_param_jac_t(self._init_grad)

    @property
    def prob(self):
        return self._prob

    @property
    def obj(self):
        return self._obj

    @property
    def backend(self):
        return self._backend

    @property
    def num_param(self):
        return self._impl.num_param

    @property
    def assembly_calls(self):
        return self._impl.assembly_calls

    @property
    def solve_calls(self):
        return self._impl.solve_calls

    @property
    def assembly_seconds(self):
        return self._impl.assembly_seconds

    @property
    def solve_seconds(self):
        return self._impl.solve_seconds

    def reset_timing(self):
        self._impl.reset_timing()

    def _check_current(self):
        if self._prob._compiled is not self._compiled:
            raise RuntimeError(
                "problem configuration changed; create a new reduced functional"
            )

    def _activate_backend(self):
        _configure_assembly(self._prob, self._backend)

    def value(self, param):
        self._check_current()
        self._activate_backend()
        return self._impl.value(self._prob._check_param(param))

    def grad(self, param):
        self._check_current()
        self._activate_backend()
        return np.asarray(self._impl.grad(self._prob._check_param(param)))

    def value_grad(self, param, progress=None):
        self._check_current()
        self._activate_backend()
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")
        value, grad = self._impl.value_grad(
            self._prob._check_param(param),
            progress=progress,
        )
        return float(value), np.asarray(grad)


class TaoOptimizer:
    """PETSc/TAO optimizer for a C++ reduced functional."""

    def __init__(self, reduced):
        if not isinstance(reduced, NavierStokesReducedFunctional):
            raise TypeError(
                "reduced must be a NavierStokesReducedFunctional"
            )
        if not hasattr(_core, "_tao_solve"):
            raise RuntimeError("femx was built without PETSc/TAO")
        self._reduced = reduced

    @property
    def reduced(self):
        return self._reduced

    def solve(
        self,
        init_param,
        *,
        bnds=None,
        max_itrs=100,
        grad_tol=1.0e-8,
        progress=None,
    ):
        if isinstance(max_itrs, bool) or not isinstance(
            max_itrs, (int, np.integer)
        ):
            raise TypeError("max_itrs must be an integer")
        if max_itrs <= 0:
            raise ValueError("max_itrs must be positive")
        grad_tol = float(grad_tol)
        if not np.isfinite(grad_tol) or grad_tol < 0.0:
            raise ValueError("grad_tol must be finite and nonnegative")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")

        self._reduced._check_current()
        self._reduced._activate_backend()
        init_param = self._reduced.prob._check_param(init_param)
        lower, upper = _bnds(bnds, self._reduced.num_param)
        result = _core._tao_solve(
            self._reduced._impl,
            init_param,
            lower=lower,
            upper=upper,
            max_itrs=int(max_itrs),
            grad_tol=grad_tol,
            progress=progress,
        )
        return TaoResult(
            param=np.asarray(result["param"], dtype=float).copy(),
            grad=np.asarray(result["grad"], dtype=float).copy(),
            obj=float(result["obj"]),
            num_iter=int(result["num_iter"]),
            num_fun=int(result["num_fun"]),
            num_grad=int(result["num_grad"]),
            converged=bool(result["converged"]),
            status=int(result["status"]),
            msg=str(result["msg"]),
        )


_BACKENDS["dense"] = (_dense_matrix, _dense_solver)
if hasattr(_core, "_ReSolveLinearSolver"):
    _BACKENDS["resolve"] = (_resolve_matrix, _resolve_solver)
if hasattr(_core, "_KspLinearSolver"):
    _BACKENDS["petsc"] = (_petsc_matrix, _petsc_solver)
