# Example: Poisson Equation

This example solves a scalar Poisson problem on the unit square:

```math
\begin{aligned}
-\Delta u &= 0
&& \text{in } \Omega = (0, 1)^2,
\\
u &= g
&& \text{on } \Gamma = \partial \Omega,
\end{aligned}
```

The boundary condition is given by

```math
\begin{aligned}
g(x, y) &=
\sin(\pi x)
&& \text{on } \Gamma_{\mathrm{top}},
\\
g(x, y) &= 0
&& \text{on } \Gamma_{\mathrm{other}}.
\end{aligned}
```

Here, $\Gamma_{\mathrm{top}}$ denotes the top boundary of the unit square, and
$\Gamma_{\mathrm{other}}$ denotes the other boundaries.

The exact solution used for verification is

```math
u_{\mathrm{exact}}(x, y) =
\frac{\sin(\pi x)\sinh(\pi y)}{\sinh(\pi)}.
```

## Run

From the build directory:

```shell
./examples/poisson/poisson --output yes
```

All variants build the finite-element operator through `Geometry`,
`AssemblyMap`, `BoundaryPlan`, and memory-space-specific CSR storage. The
default executable uses the native dense fallback to solve that assembled CSR
system and does not require optional solver packages.

With Re::Solve enabled:

```shell
./examples/poisson/poisson-resolve --nx 32 --ny 32 -b cpu --output yes
```

With a CUDA-enabled Re::Solve build, `-b cuda` keeps geometry, maps/plans, CSR
values, right-hand side, and solution in device memory through assembly and
the linear solve. Only the final report/output solution is copied to the host:

```shell
./examples/poisson/poisson-resolve --nx 32 --ny 32 -b cuda --output yes
```

With PETSc enabled:

```shell
./examples/poisson/poisson-petsc --nx 32 --ny 32 --output yes
```
