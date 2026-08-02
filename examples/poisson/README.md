# Example: Poisson Equation

This example solves a scalar Poisson problem on the unit square. It
demonstrates the standard finite-element workflow: defining a Q1
discretization, evaluating element integrals, assembling the global system,
applying Dirichlet boundary conditions, and solving the resulting linear
system.

## Problem setting

Let

```math
\Omega=(0,1)^2,
\qquad
\Gamma_{\mathrm{top}}=\{(x,1):0\leq x\leq1\},
\qquad
\Gamma_{\mathrm{other}}=\partial\Omega\setminus\Gamma_{\mathrm{top}}.
```

The boundary-value problem is

```math
\begin{aligned}
-\Delta u &= 0
&& \text{in } \Omega,
\\
u &= \sin(\pi x)
&& \text{on } \Gamma_{\mathrm{top}},
\\
u &= 0
&& \text{on } \Gamma_{\mathrm{other}}.
\end{aligned}
```

The exact solution used for verification is

```math
u_{\mathrm{exact}}(x,y)
=
\frac{\sin(\pi x)\sinh(\pi y)}{\sinh(\pi)}.
```

## Finite-element discretization

The unit square is divided into `nx` by `ny` axis-aligned quadrilateral
elements. The solution is approximated in the continuous bilinear Q1
finite-element space $V_h$. There is one degree of freedom per mesh node,
giving `(nx + 1) * (ny + 1)` unknowns.

The Dirichlet boundary conditions are applied at boundary nodes. The discrete
weak problem seeks $u_h \in V_h$ satisfying those prescribed values such that

```math
\int_\Omega \nabla u_h\cdot\nabla v_h\,d\Omega=0
\qquad
\forall v_h\in V_h.
```

For local Q1 basis functions $N_i$, `PoissonElementKernel` evaluates each
element stiffness entry as

```math
A^e_{ij}
\approx
\sum_{q=1}^{4}
\nabla N_i(x_q)\cdot\nabla N_j(x_q)\,
\left|\det J(x_q)\right|w_q.
```

The four points are the tensor-product 2-by-2 Gauss rule on the reference quadrilateral. Here, $w_q$ is the quadrature weight and $J(x_q)$ is the Jacobian of the mapping from the reference element to the physical element.

The element matrices $A^e$ are assembled into the global stiffness matrix $A$. For each Dirichlet boundary node $i$, row $i$ of $A$ is replaced by an identity row and $b_i=g_i$. This gives

```math
Ax=b.
```

Here, $x$ contains the nodal solution values.
`AssemblyMap` maps the element entries into the global CSR matrix, while `BoundaryMap` identifies the rows to replace.

The femx solver interface expresses this system as $R(x)=Ax-b=0$.
`LinearStateSolver` assembles $R(0)=-b$ and $A=dR/dx$, then solves $Ax=b$.

## Executables and backends

| Executable | Linear solver | Supported execution |
| --- | --- | --- |
| `poisson` | Built-in dense fallback | CPU |
| `poisson-resolve` | Re::Solve | CPU, or CUDA when enabled in the Re::Solve build |
| `poisson-petsc` | PETSc | CPU; invoked collectively on `PETSC_COMM_WORLD` |

The dense executable does not require an optional solver package. For the
CUDA Re::Solve path, element assembly, CSR values, right-hand side, solution,
and the linear solve remain in device memory. Only the final solution used
for reporting and output is copied to the host. In a PETSc run, only the root
MPI rank prints the report and writes the VTU file.

## Run

The common options are:

| Option | Default | Meaning |
| --- | --- | --- |
| `--nx N` | `8` | Number of cells in the x direction. |
| `--ny N` | `8` | Number of cells in the y direction. |
| `-b BACKEND`, `--backend BACKEND` | `cpu` | Select `cpu` or `cuda`, subject to the executable restrictions above. |
| `--output VALUE` | `no` | Pass `yes` to write a VTU file in the example's build-local `output` directory. |

From the build directory, run the built-in dense version with

```shell
./examples/poisson/poisson --nx 32 --ny 32 --output yes
```

With Re::Solve enabled, run on the CPU with

```shell
./examples/poisson/poisson-resolve --nx 32 --ny 32 -b cpu --output yes
```

With a CUDA-enabled Re::Solve build, run the same discrete problem on the GPU
with

```shell
./examples/poisson/poisson-resolve --nx 32 --ny 32 -b cuda --output yes
```

With PETSc enabled, run with

```shell
./examples/poisson/poisson-petsc --nx 32 --ny 32 --output yes
```

Standard PETSc options may be appended to the `poisson-petsc` command line.
