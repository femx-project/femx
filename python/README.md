# femx Python API

Python bindings for the femx C++ implementation.

## Install

Install femx from the repository root:

```shell
python3 -m pip install .
```

## Navier–Stokes example

```python
import femx


def lid_velocity(point, time):
    x = point[0]
    return 1.0 - 4.0 * x * x, 0.0


model = femx.NavierStokesModel(
    "data/meshes/2d_rectangle_res_very_low.msh",
    num_steps=50,
    dt=0.02,
    rho=1.0,
    mu=0.01,
)

problem = femx.NavierStokesProblem(model)
problem.add_bc(femx.DirichletBC("wall_top", "velocity", lid_velocity))
for wall in ("wall_left", "wall_right", "wall_bottom"):
    problem.add_bc(femx.DirichletBC(wall, "velocity", (0.0, 0.0)))
problem.build()

trajectory = problem.solve(backend="petsc")
model.write_xdmf("cavity.xdmf", trajectory)
```

The output can be opened in ParaView.
