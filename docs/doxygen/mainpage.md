# femx API {#mainpage}

femx is a finite-element research code for PDE-constrained forward and inverse
problems.  The public API is organized around a few core layers:

- `%femx::fem` provides meshes, finite-element spaces, elements, quadrature,
  and boundary utilities.
- `%femx::assembly` provides residual, Jacobian, and sparse-matrix assembly.
- `%femx::state` provides steady and time-dependent state solvers.
- `%femx::inverse` provides objectives, observations, regularization, and
  reduced functionals.
- `%femx::linalg` provides Host/CUDA, PETSc, and ReSolve linear-system
  implementations.
- `%femx::io` provides lightweight visualization and time-series I/O.
- `%femx::opt` provides optimization interfaces.
- `%femx::runtime` provides runtime, CLI, and parallel-execution utilities.
- `%femx::model::navier` provides the Navier--Stokes forward model.

For v0.7.0, start with the C++ Poisson examples or the supported Python Navier
example for end-to-end usage. This release aligns public function names with
the project naming conventions and reduces preprocessing time and memory for
time-dependent Dirichlet conditions.
