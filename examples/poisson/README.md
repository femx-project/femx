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

The default executable uses the native dense solver and does not require
optional solver packages.

With Re::Solve enabled:

```shell
./examples/poisson/poisson-resolve --nx 32 --ny 32 -b cpu --output yes
```

With PETSc enabled:

```shell
./examples/poisson/poisson-petsc --nx 32 --ny 32 --output yes
```
