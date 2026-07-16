#!/usr/bin/env python3
"""Solve two-dimensional channel flow with PETSc."""

from pathlib import Path

import femx


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "python/runs/navier_stokes"


def inlet_velocity(point, _time):
    y = point[1] / 0.002
    return 1.0 - y * y, 0.0


def print_progress(event):
    print(
        f"\rStep {event['step']}/{event['total']}",
        end="",
        flush=True,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = femx.NavierStokesModel(
        ROOT / "data/meshes/2d_rectangle.msh",
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

    model.write_xdmf(OUTPUT_DIR / "flow.xdmf", trajectory)


if __name__ == "__main__":
    main()
