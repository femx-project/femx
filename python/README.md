# femx Python API

The supported Python package exposes coarse-grained finite-element operations
from the femx C++ engine. The standard build requires the ReSolve 0.99.2
development API at commit `bd60de0` or later; the `v0.99.2` tag is too old for
the adapter despite reporting the same version. PETSc 3.19 or later and MPI are
also required. Install the solver libraries, then install femx from the
repository root. `petsc4py` is not required:

```shell
python3 -m pip install .
```

The first supported vertical slice provides mesh inspection and scalar P1
operators on named or tagged simplicial boundary surfaces:

```python
import femx

mesh = femx.Mesh.read("data/meshes/2d_straighttube.msh")
inlet = mesh.boundary("inlet")

K, M, load = inlet.laplacian_matrices()
profile = inlet.poisson_profile()
eigenvalues, modes = inlet.laplacian_modes(3)
```

`profile` solves the surface Poisson problem with homogeneous Dirichlet values
on the boundary-surface rim and is normalized to unit surface integral. The
columns of `modes` are the lowest Dirichlet surface-Laplacian modes and are
orthonormal in the `M` inner product.

Run the Python tests from an installed environment with:

```shell
python3 -m unittest discover -s python/tests -v
```

The older `femx_experimental` module remains available behind the separate
`FEMX_BUILD_PYTHON_EXPERIMENTAL` CMake option while clients migrate.

## Navier-Stokes model

`NavierStokesModel` owns the reusable C++ discretization without choosing
which boundaries are inverse controls:

```python
model = femx.NavierStokesModel(
    "mesh.msh", num_steps=100, dt=0.01, rho=1.0, mu=0.004
)

all_velocity_dofs = model.velocity_dofs
inlet_velocity_dofs = model.velocity_boundary_dofs("inlet")
base_residual = model.residual
```

Fixed Dirichlet conditions are configured by `NavierStokesProblem`. A boundary
can be selected by its physical name or tag:

```python
prob = femx.NavierStokesProblem(model)
prob.add_bc(
    femx.DirichletBC(
        boundary="wall",
        field="velocity",
        value=(0.0, 0.0),
    )
)
prob.add_bc(
    femx.DirichletBC(
        boundary=4,
        field="velocity",
        value=lambda point, time: (inlet_speed(point, time), 0.0),
    )
)
prob.add_bc(
    femx.DirichletBC(
        boundary="outlet",
        field="pressure",
        value=0.0,
    )
)
prob.build()

residual = prob.residual
initial_state = prob.initial_state
param = prob.param
```

Use `prob.set_initial_state(state)` before `build()` to supply a fixed initial
state. This does not add optimization parameters. The state must preserve the
fixed Dirichlet values.

Callable values are sampled once at every boundary node and solve time level
during `build()`; the nonlinear solve does not call back into Python. If no
pressure condition is supplied, the problem automatically fixes one pressure
degree of freedom to zero. Parameter layouts, inverse-control boundaries,
matrices, and linear solvers remain separate higher-level concerns.

The sampling, duplicate-value checks, time-level layout, and initial-state
construction are implemented by the C++ `makeTimeDirichletData` utility.
The C++ forward problem and this Python interface therefore use the same fixed
Dirichlet compilation path.

## Velocity observations

Velocity observations use the C++ point interpolator for spatial sampling and
support arbitrary times between solve levels:

```python
op = femx.VelocityObservationOperator(
    model,
    points=((0.01, 0.0), (0.02, 0.0)),
    times=(0.1, 0.2),
    components=(0,),
)
pred = op.predict(traj)
state_direction = op.apply_transpose(obs_direction)

obs = femx.GaussianObservation(
    op,
    values=measured_velocity,
    std=measurement_sigma,
)
obj = obs.objective(num_param=prob.num_param)
misfit = obj.value(traj, param)
state_grad = obs.state_grad(traj, param)

reduced = prob.create_reduced(obj, backend="dense")
misfit, ctr_grad = reduced.value_grad(param)
```

`femx.solver_backends()` reports the backends registered in the current build.
`create_solver()`, `create_reduced()`, and `solve()` use the same backend names.
Use `backend="resolve"` for the ReSolve CPU workspace. A CUDA-enabled ReSolve
build also registers `backend="resolve-cuda"`. The CSR matrix and ReSolve
solver objects remain internal to these coarse-grained APIs.
Use `backend="petsc"` for PETSc KSP. PETSc is initialized lazily when this
backend is first used; dense and ReSolve runs do not initialize MPI. Under
`mpiexec`, the mesh and state remain replicated, while element assembly is
partitioned between ranks and the PETSc matrix, vectors, and KSP use
`PETSC_COMM_WORLD`. `femx.petsc_rank()`, `femx.petsc_size()`,
`femx.petsc_barrier()`, and `femx.petsc_finalize()` expose the small amount of
PETSc runtime state needed by application entry points. Call `petsc_finalize()`
collectively after PETSc-backed objects have been released.

PETSc/TAO can optimize the same C++ reduced functional without a Python
callback at each objective evaluation:

```python
reduced = prob.create_reduced(obj, backend="petsc")
result = femx.TaoOptimizer(reduced).solve(
    init_param,
    bnds=[(0.0, None)] * prob.num_param,
    max_itrs=100,
    grad_tol=1.0e-8,
)
```

The optimizer uses `PETSC_COMM_WORLD`, LMVM for unconstrained problems, and
BLMVM when bounds are supplied.

The complete velocity and pressure trajectory can be opened directly in
ParaView through an XDMF file:

```python
traj = prob.solve(param, backend="dense")
model.write_xdmf("results/flow.xdmf", traj)
```

This writes `results/flow.xdmf` and `results/flow.h5` using the femx C++
time-series writer.

`apply_transpose()` returns one state-space vector per solve level. The C++
least-squares objective supports observation-specific standard deviations and
time interpolation. The reduced functional solves the forward and adjoint
problems to return the gradient with respect to the inlet control parameters.
Without Enzyme, Navier-Stokes history Jacobians use an element-local central
difference fallback intended for small dense runs and gradient verification.

## Time integration

The initial state API exposes the abstract `TimeResidual` contract and the
concrete `TimeLinearIntegrator`. A Python residual returns NumPy arrays while
the time loop, matrix assembly, and linear solve remain in C++:

```python
import numpy as np
import femx

class Recurrence(femx.TimeResidual):
    def __init__(self):
        super().__init__()

    def dims(self):
        dims = femx.TimeDims()
        dims.num_steps = 3
        dims.num_states = 1
        dims.num_param = 1
        dims.num_residuals = 1
        return dims

    def residual(self, context):
        return (context["next_state"] - context["history"][0]
                - context["parameters"])

    def apply_jacobian(self, context, variable, direction):
        return direction if variable.is_next_state else -direction

    def apply_jacobian_transpose(self, context, variable, adjoint):
        return adjoint if variable.is_next_state else -adjoint

    def assemble_jacobian(self, context, variable):
        return np.array([[1.0 if variable.is_next_state else -1.0]])

problem = Recurrence()
matrix = femx.DenseAssemblyMatrix()
linear_solver = femx.DenseLinearSolver()
integrator = femx.TimeLinearIntegrator(problem, matrix, linear_solver)
integrator.set_initial_state(np.array([1.0]))

trajectory = integrator.solve(np.array([2.0]))
values = np.asarray(trajectory)  # zero-copy (time level, state dof)
```

`TimeLinearIntegrator` retains its Python arguments for its entire lifetime,
matching the non-owning references used by the C++ class. The dense backend is
portable and useful for small problems and tests. ReSolve uses the same
`AssemblyMatrix` and `LinearSolver` interfaces with sparse CSR storage.
