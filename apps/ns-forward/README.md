## App: Navier–Stokes Equations

This solver uses a Galerkin/least-squares (GLS) stabilized finite element formulation for the unsteady incompressible Navier-Stokes equations.

The governing equations are written as

```math
\begin{aligned}
\rho \frac{\partial \mathbf{u}}{\partial t}
+
\rho (\mathbf{u} \cdot \nabla)\mathbf{u}
+
\nabla p
-
\mu \nabla^2 \mathbf{u}
&=
\mathbf{0}
&& \quad \text{in } \Omega \times (0,T],
\\
\nabla \cdot \mathbf{u}
&=
0
&& \quad \text{in } \Omega \times (0,T].
\end{aligned}
```

where $\mathbf{u}$ is the velocity, $p$ is the pressure, $\rho$ is the density, and $\mu$ is the dynamic viscosity.

Let $\mathbf{w}$ and $q$ be the test functions for velocity and pressure,
respectively.

For time integration, the transient term is discretized by

```math
\frac{\partial \mathbf{u}}{\partial t}
\approx
\frac{\mathbf{u}^{n+1} - \mathbf{u}^{n}}{\Delta t}.
```

The nonlinear advection term is treated by separating the convecting
velocity from the transported velocity. The convecting velocity is
extrapolated with Adams-Bashforth,

```math
\mathbf{u}_{\mathrm{adv}}^{n+1/2}
=
\frac{3}{2}\mathbf{u}^{n}
-
\frac{1}{2}\mathbf{u}^{n-1},
```

At the first time step, where no previous step is available, the code uses
$\mathbf{u}_{\mathrm{adv}}=\mathbf{u}^{n}$.

The transported velocity in the advection term and the diffusion term are
evaluated by Crank-Nicolson,

```math
\mathbf{u}^{n+1/2}
=
\frac{1}{2}
\left(
\mathbf{u}^{n+1}
+
\mathbf{u}^{n}
\right).
```

The time-discrete Galerkin contribution is

```math
\begin{aligned}
F_{\mathrm{G}}^{n+1/2}
=&
\int_{\Omega}
\left[
\rho
\mathbf{w}
\cdot
\frac{\mathbf{u}^{n+1}-\mathbf{u}^{n}}{\Delta t}
+
\rho
\mathbf{w}
\cdot
\left(
\mathbf{u}_{\mathrm{adv}}^{n+1/2}
\cdot
\nabla
\right)
\mathbf{u}^{n+1/2}
-
p^{n+1}
\nabla \cdot \mathbf{w}
\right] d\Omega
\\
&+
\int_{\Omega}
\left[
\mu
\nabla \mathbf{w}
:
\nabla \mathbf{u}^{n+1/2}
+
q
\nabla \cdot \mathbf{u}^{n+1}
\right] d\Omega .
\end{aligned}
```

The GLS stabilization contribution is

```math
\begin{aligned}
F_{\mathrm{GLS}}^{n+1/2}
=&
\sum_{e=1}^{N_e}
\int_{\Omega_e}
\frac{1}{\rho}
\left[
\tau_m
\rho
\left(
\mathbf{u}_{\mathrm{adv}}^{n+1/2}
\cdot
\nabla
\right)
\mathbf{w}
+
\tau_p
\nabla q
\right]
\cdot
\left[
\rho
\frac{\mathbf{u}^{n+1} - \mathbf{u}^{n}}{\Delta t}
+
\rho
\left(
\mathbf{u}_{\mathrm{adv}}^{n+1/2}
\cdot
\nabla
\right)
\mathbf{u}^{n+1/2}
+
\nabla p^{n+1}
\right]
\, d\Omega
.
\end{aligned}
```

The terms multiplied by $\tau_m$ and $\tau_p$ are the SUPG and PSPG
contributions, respectively. The factor $1/\rho$ follows the scaling used by
Tezduyar [1]. With this scaling, the SUPG part is equivalent to
$\tau_m(\mathbf{u}_{\mathrm{adv}}\cdot\nabla)\mathbf{w}$ dotted with the
momentum residual, while the PSPG part is equivalent to
$(\tau_p/\rho)\nabla q$ dotted with the same residual. The diffusion part and
body-force term are not included in the bracketed momentum expression. The full
discrete weak form is

```math
F^{n+1/2}
=
F_{\mathrm{G}}^{n+1/2}
+
F_{\mathrm{GLS}}^{n+1/2}
=
0.
```

The stabilization parameter is computed element-wise as

```math
\tau_m
=
\tau_p
=
\left[
\left(\frac{2}{\Delta t}\right)^2
+
\left(\frac{2\lVert\mathbf{u}\rVert}{h}\right)^2
+
\left(\frac{4\nu}{h^2}\right)^2
\right]^{-1/2},
\qquad
\nu = \frac{\mu}{\rho}.
```

This is a local length-scale approximation of the SUPG/PSPG stabilization
parameter calculations described by Tezduyar [1], with $\tau_m=\tau_p$ and
$h$ used as the element length. The velocity norm in this expression is
evaluated from the current history state used to assemble the time-step
operator.

The implementation assembles the time-step residual in affine form,
$R(\mathbf{x}^{n+1})=K\mathbf{x}^{n+1}-F$, and solves the resulting linear
system $K\mathbf{x}^{n+1}=F$ once per time step. It does not perform a Newton
iteration on the fully nonlinear Navier-Stokes residual.

The model publishes flattened element data and an immutable `AssemblyMap`.
Both app entry points allocate their matrix from the map's CSR pattern. PETSc
uses Host assembly and exact distributed preallocation. In a CUDA ReSolve
build, the `resolve` entry point keeps the trajectory, residual, Jacobian,
boundary work, and linear solve on Device and assembles through
`CudaAssembly.hpp`. Both paths evaluate the same Navier--Stokes row operator.

Boundary conditions are applied in configuration order. A later condition
replaces an earlier value at a shared node, so list wall conditions after
inlets, outlets, or a moving cavity lid when the wall should own the rim.

## Run

From your build directory, run the executable for the backend you enabled:

```shell
./apps/ns-forward/ns-forward-resolve \
  --config ../apps/ns-forward/configs/resolve/cavity/Config.json

./apps/ns-forward/ns-forward-petsc \
  --config ../apps/ns-forward/configs/petsc/cavity/Config.json
```

## References

1. T. E. Tezduyar, "Calculation of the Stabilization Parameters in SUPG and PSPG Formulations," 2002. https://www.tafsm.org/PUB_PRE/ipALL/ip90-MECOM02-SP.pdf
