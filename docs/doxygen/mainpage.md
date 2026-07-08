# femx API {#mainpage}

femx is a finite-element research code for PDE-constrained forward and inverse
problems.  The public API is organized around a few core layers:

- `femx::Mesh`, `femx::FESpace`, and finite elements describe the discrete
  geometry and degrees of freedom.
- `femx::assembly` classes assemble residuals, objectives, and sparse matrices
  from element-local kernels.
- `femx::state` defines residual interfaces and provides state solvers.
- `femx::inverse` provides objective, observation, regularization, and
  reduced-functional utilities.
- `femx::linalg` adapts PETSc, ReSolve, and native linear algebra backends.
- `femx::io::VtuWriter` writes lightweight VTU visualization files.

For v0.1.0, the Poisson examples are the best starting point for end-to-end
usage.
