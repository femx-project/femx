# femx API {#mainpage}

femx is a finite-element research code for PDE-constrained forward and inverse
problems.  The public API is organized around a few core layers:

- `%femx::fem` provides meshes, finite-element spaces, elements, quadrature,
  and boundary utilities.
- `%femx::assembly` provides residual, Jacobian, and sparse-matrix assembly.
- `%femx::state` provides steady and time-dependent state solvers.
- `%femx::inverse` provides objectives, observations, regularization, and
  reduced functionals.
- `%femx::linalg` provides native, PETSc, and ReSolve linear algebra backends.
- `%femx::io` provides lightweight visualization and time-series I/O.
- `%femx::opt` provides optimization interfaces.
- `%femx::runtime` provides runtime, CLI, and parallel-execution utilities.
- `%femx::model::ns` provides the Navier--Stokes forward model.

For v0.1.0, the Poisson examples are the best starting point for end-to-end
usage.
