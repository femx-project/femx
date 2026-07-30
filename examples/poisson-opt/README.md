# Example: Poisson Boundary Control

This example estimates a Dirichlet control on the upper boundary of the unit
square:

```math
\begin{aligned}
-\Delta u &= 0 && \text{in } \Omega=(0,1)^2, \\
u &= m && \text{on } \Gamma_{\mathrm{top}}, \\
u &= 0 && \text{on } \Gamma_{\mathrm{other}}.
\end{aligned}
```

The control vector `m` contains the control values at the top boundary nodes.
Synthetic observations are generated from

```math
u_{\mathrm{exact}}(x,y)
=
\frac{\sin(2\pi x)\sinh(2\pi y)}{\sinh(2\pi)}.
```

TAO minimizes the reduced objective

```math
\frac{1}{2}\sum_{i\in\mathcal O}
w_i\left(u_i(m)-d_i\right)^2
+
\frac{\alpha}{2}\int_{\Gamma_{\mathrm{top}}}m^2\,d\Gamma.
```

The initial control is zero. `--obs-stride 0` selects an observation spacing
equal to one eighth of the smaller mesh dimension.

`PoissonOptProblem` contains only the mathematical and discretization data.
`HostPoissonOptResidual` and `DevicePoissonOptResidual` implement the same
controlled PDE in different memory spaces. `optimize()` is shared by both executables.

## Backends

| Executable | State/adjoint solve | Execution | TAO communicator |
| --- | --- | --- | --- |
| `poisson-opt-resolve` | ReSolve | `-b cpu` | serial |
| `poisson-opt-resolve` | ReSolve | `-b cuda` | CUDA |
| `poisson-opt-petsc` | PETSc | CPU, distributed assembly | `PETSC_COMM_WORLD` |

TAO is provided by PETSc in all three cases. In an MPI PETSc run, every rank
participates in the TAO callbacks, finite-element assembly, forward solve, and
adjoint solve. Only rank zero prints the report and writes visualization files.

## Build

PETSc version (ReSolve is not required):

```shell
cmake --preset petsc
cmake --build --preset petsc --target poisson-opt-petsc
```

ReSolve version (PETSc provides TAO):

```shell
cmake --preset resolve-petsc \
  -DReSolve_DIR=/path/to/resolve/lib/cmake/resolve
cmake --build --preset resolve-petsc --target poisson-opt-resolve
```

Both versions use the analytic parameter derivative by default. To use Enzyme,
select the `petsc-enzyme` or `resolve-petsc-enzyme` preset and set `Enzyme_DIR`.
CUDA Enzyme also requires Clang as both the C++ and CUDA compiler.
The CUDA Enzyme path is experimental.

## Run

ReSolve on the CPU:

```shell
./build/resolve-petsc/examples/poisson-opt/poisson-opt-resolve \
  -b cpu --nx 32 --ny 32 --max-its 50
```

ReSolve on CUDA:

```shell
./build/resolve-petsc/examples/poisson-opt/poisson-opt-resolve \
  -b cuda --nx 32 --ny 32 --max-its 50
```

PETSc and TAO on four MPI ranks:

```shell
mpiexec -n 4 \
  ./build/petsc/examples/poisson-opt/poisson-opt-petsc \
  -b cpu --nx 32 --ny 32 --max-its 50
```

Pass `--output yes` to write the final mesh fields and observation point cloud
under the example's build-local `output` directory. Standard PETSc and TAO
options can be added to either executable.

## Result

<p align="center">
  <img src="../../docs/figs/poisson-opt.png" alt="Poisson boundary-control optimization result" width="560">
</p>
