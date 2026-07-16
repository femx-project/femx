# femx Python API

Python bindings for the femx C++ implementation.

## Install

Install femx from the repository root:

```shell
python -m pip install .
```

## Navier–Stokes example

Run the example from the repository root:

```shell
python python/examples/navier_stokes.py
```

```python
from pathlib import Path

import femx


def inlet_velocity(point, _time):
    y = point[1] / 0.002
    return 1.0 - y * y, 0.0


def print_progress(event):
    print(
        f"\rStep {event['step']}/{event['total']}",
        end="",
        flush=True,
    )


model = femx.NavierStokesModel(
    "data/meshes/2d_rectangle.msh",
    num_steps=100,
    dt=1.0e-4,
    rho=1.0,
    mu=4.0e-5,
)

problem = femx.NavierStokesProblem(model)
problem.add_bc(femx.DirichletBC("wall", "velocity", (0.0, 0.0)))
problem.add_bc(femx.DirichletBC("inlet", "velocity", inlet_velocity))
problem.add_bc(femx.DirichletBC("outlet", "pressure", 0.0))
problem.build()

solver = problem.create_solver(backend="petsc")
trajectory = solver.solve(progress=print_progress)
print()

output_dir = Path("python/runs/navier_stokes")
output_dir.mkdir(parents=True, exist_ok=True)
model.write_xdmf(output_dir / "flow.xdmf", trajectory)
```

This setup has a Reynolds number of 100 and a maximum CFL of about 0.64.
The output in `python/runs/navier_stokes` can be opened in ParaView.
