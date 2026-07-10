# Poisson Optimization Example

This example estimates an upper-boundary Dirichlet control for a scalar
Poisson problem on the unit square:

```math
\begin{aligned}
-\Delta u &= 0
&& \text{in } \Omega = (0, 1)^2,
\\
u &= m
&& \text{on } \Gamma_{\mathrm{top}},
\\
u &= 0
&& \text{on } \Gamma_{\mathrm{other}}.
\end{aligned}
```

Here, $m$ is the unknown boundary control, $\Gamma_{\mathrm{top}}$ denotes the
top boundary of the unit square, and $\Gamma_{\mathrm{other}}$ denotes the
other boundaries.

The target state is generated from

```math
u_{\mathrm{exact}}(x, y) =
\frac{\sin(2\pi x)\sinh(2\pi y)}{\sinh(2\pi)}.
```

The optimization problem minimizes

```math
\min_m
\frac{1}{2}
\sum_{i \in \mathcal{O}}
w_i
\left(u_i(m) - d_i\right)^2
+
\frac{\alpha}{2}
\int_{\Gamma_{\mathrm{top}}} m^2 \, d\Gamma,
```

where $d_i$ are sparse observations and $\alpha$ controls the regularization
strength.

## Run

From the build directory:

```shell
./examples/poisson-opt/poisson-opt-petsc --nx 48 --ny 48 --output yes --max-its 50
```

The optimization driver uses PETSc/TAO.

With Re::Solve enabled for the forward and adjoint linear solves:

```shell
./examples/poisson-opt/poisson-opt-resolve --nx 48 --ny 48 -b cpu --output yes --max-its 50
```
