# femx Python API

Python bindings for the femx finite-element engine.

## Install

Install femx from the repository root:

```shell
python3 -m pip install .
```

## Navier–Stokes example

```python
import femx


def inlet_velocity(point, time):
    y = point[1] / 0.01
    return 4.0 * y * (1.0 - y), 0.0


model = femx.NavierStokesModel(
    "data/meshes/2d_straighttube.msh",
    num_steps=100,
    dt=0.01,
    rho=1.0,
    mu=0.004,
)

problem = femx.NavierStokesProblem(model)
problem.add_bc(femx.DirichletBC("wall", "velocity", (0.0, 0.0)))
problem.add_bc(femx.DirichletBC("inlet", "velocity", inlet_velocity))
problem.add_bc(femx.DirichletBC("outlet", "pressure", 0.0))
problem.build()

trajectory = problem.solve(backend="dense")
model.write_xdmf("results/flow.xdmf", trajectory)
```

The output can be opened in ParaView.
