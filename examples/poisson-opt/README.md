# Example: Poisson Boundary Control

This example recovers an unknown Dirichlet boundary condition from sparse
observations inside the domain. It demonstrates a PDE-constrained
optimization workflow with forward and adjoint Poisson solves.

## Problem setting

Let

```math
\Omega=(0,1)^2,
\qquad
\Gamma_{\mathrm{control}}=\{(x,1):0<x<1\},
\qquad
\Gamma_{\mathrm{fixed}}=\partial\Omega\setminus
\Gamma_{\mathrm{control}}.
```

For a boundary control $m$, the state $u(m)$ satisfies

```math
\begin{aligned}
-\Delta u &= 0
&& \text{in }\Omega,
\\
u &= m
&& \text{on }\Gamma_{\mathrm{control}},
\\
u &= 0
&& \text{on }\Gamma_{\mathrm{fixed}}.
\end{aligned}
```

The two top-corner values remain zero. A manufactured target control

```math
m_{\mathrm{target}}(x)=\sin(2\pi x)
```

corresponds to the harmonic target solution

```math
u_{\mathrm{target}}(x,y)
=
\frac{\sin(2\pi x)\sinh(2\pi y)}{\sinh(2\pi)}.
```

The target control is sampled at the controlled nodes and the discrete
Poisson equation is solved once. Values of that solution at selected interior
nodes become the noise-free observations $d_i$ used by the optimization.

## Finite-element discretization

The unit square is divided into `nx` by `ny` axis-aligned quadrilateral
elements. As in the [forward Poisson example](../poisson/README.md), the state
is approximated with continuous bilinear Q1 finite elements, with one degree
of freedom per mesh node and a 2-by-2 Gauss rule on each element.

The element matrices $A^e$ are assembled into the global stiffness matrix
$A$. For each Dirichlet boundary node $i$, row $i$ of $A$ is replaced by an
identity row. The right-hand side is $m_j$ at a controlled top node and zero
at all other nodes. Thus the discrete state equation is

```math
A x(m)=b(m),
```

where $x(m)$ contains the nodal state values.

Interior observations are selected on a regular subset of mesh nodes. If the
observation stride is $s$, every $s$-th interior node is observed in each
coordinate direction. With `--obs-stride 0`, the spacing is selected
automatically. For example, a `32` by `32` mesh uses a stride of `4`, giving
`49` observation nodes.

The reduced objective is

```math
J(m)
=
\frac{1}{2N_{\mathrm{obs}}}
\sum_{i\in\mathcal O}\left(x_i(m)-d_i\right)^2
+
\frac{\alpha}{2}\sum_j h_x m_j^2,
\qquad
h_x=\frac{1}{n_x}.
```

The first term measures the mismatch at the $N_{\mathrm{obs}}$ observation
nodes. The second is a regularization term that prevents $m$ from becoming
unnecessarily large. The default regularization weight is $\alpha=10^{-6}$.

## Optimization workflow

1. Solve the state equation at $m_{\mathrm{target}}$ to generate the
   observations.
2. Initialize the control to zero.
3. For each TAO objective and gradient evaluation, solve the forward equation
   for $x(m)$ and an adjoint equation with $A^T$.
4. Use the adjoint solution to evaluate the gradient of $J(m)$, then let TAO
   update the control.
5. After TAO stops, solve the state equation once more at the optimized
   control for reporting and visualization.

The femx solver interface expresses the state equation as
$R(x,m)=Ax-b(m)=0$. `HostPoissonOptResidual` and
`CudaPoissonOptResidual` assemble the same equation in host and device memory,
respectively. The shared `optimize()` routine performs the target, forward,
adjoint, and final-state solves described above.

The report includes the final objective and gradient norm. State errors are
computed at all mesh nodes against the discrete target state, and control
errors are computed at the controlled top-boundary nodes against
$m_{\mathrm{target}}$.

## Backends

| Executable | Forward/adjoint solver | Execution | TAO communicator |
| --- | --- | --- | --- |
| `poisson-opt-resolve` | Re::Solve | CPU | `PETSC_COMM_SELF` |
| `poisson-opt-resolve` | Re::Solve | CUDA | `PETSC_COMM_SELF` |
| `poisson-opt-petsc` | PETSc | CPU with distributed matrices | `PETSC_COMM_WORLD` |

TAO is provided by PETSc for both executables. On the CUDA path, TAO updates
the control on the host while the forward and adjoint systems are assembled
and solved on the device. In an MPI PETSc run, every rank participates in the
optimization and linear solves; only rank zero prints the report and writes
visualization files.

## Build

Build the PETSc version without Re::Solve using

```shell
cmake --preset petsc
cmake --build --preset petsc --target poisson-opt-petsc
```

Build the Re::Solve version using

```shell
cmake --preset resolve-petsc \
  -DReSolve_DIR=/path/to/resolve/lib/cmake/resolve
cmake --build --preset resolve-petsc --target poisson-opt-resolve
```

The analytic control derivative is used by default. To use Enzyme, select the
`petsc-enzyme` or `resolve-petsc-enzyme` preset and set `Enzyme_DIR`. CUDA
Enzyme also requires Clang as both the C++ and CUDA compiler and is
experimental.

## Run

The example options are:

| Option | Default | Meaning |
| --- | --- | --- |
| `--nx N` | `32` | Number of cells in the x direction. |
| `--ny N` | `32` | Number of cells in the y direction. |
| `-b BACKEND`, `--backend BACKEND` | `cpu` | Select `cpu` or `cuda`, subject to the backend restrictions above. |
| `--alpha A` | `1e-6` | Set the nonnegative regularization weight. |
| `--obs-stride N` | `0` | Set the observation spacing in mesh cells; `0` selects it automatically. |
| `--max-its N` | `50` | Set the maximum number of TAO iterations. |
| `--output VALUE` | `no` | Pass `yes` to write VTU files in the example's build-local `output` directory. |

Run Re::Solve on the CPU with

```shell
./build/resolve-petsc/examples/poisson-opt/poisson-opt-resolve \
  -b cpu --nx 32 --ny 32 --max-its 50
```

Run Re::Solve on CUDA with

```shell
./build/resolve-petsc/examples/poisson-opt/poisson-opt-resolve \
  -b cuda --nx 32 --ny 32 --max-its 50
```

Run PETSc and TAO on four MPI ranks with

```shell
mpiexec -n 4 \
  ./build/petsc/examples/poisson-opt/poisson-opt-petsc \
  -b cpu --nx 32 --ny 32 --max-its 50
```

Standard PETSc and TAO options may be appended to any command above.

## Output

Pass `--output yes` to write two files:

- `poisson-opt-nxN-nyN.vtu` contains `state`, `target_state`, `state_error`,
  `control`, `target_control`, `control_error`, and `control_mask` on the mesh.
- `poisson-opt-nxN-nyN.observations.vtu` contains `observation`, `prediction`,
  `misfit`, and `weight` on the observation point cloud.

<p align="center">
  <img src="../../docs/figs/poisson-opt.png" alt="Poisson boundary-control optimization result" width="560">
</p>
