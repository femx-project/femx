# femx

femx is a small C++ finite element library for experimenting with sparse linear solves.

The library currently provides core types, mesh, finite element, assembly,
linear algebra, system backend, equation-solving, boundary-condition, I/O, and
inverse-problem components. It also includes examples and forward
applications, including an unsteady Navier-Stokes GLS solver.

## Getting started

Dependencies:

- CMake >= 3.22
- C++17 compiler
- ReSolve (optional, for the linear solver backend)
- HDF5 (optional, for HDF5/XDMF output)
- OpenMP (optional, for parallel assembly)
- PETSc and MPI (optional, for the PETSc Navier-Stokes GLS application)
- Enzyme (optional, for automatic differentiation)

To build the library, examples and applications:

```shell
$ cmake -S . -B build
$ cmake --build build
```

## CMake options

Common configuration options are:

- `FEMX_ENABLE_RESOLVE=ON`: enable the ReSolve linear solver backend.
- `FEMX_ENABLE_HDF5=ON`: enable HDF5/XDMF output support.
- `FEMX_ENABLE_OPENMP=ON`: enable OpenMP parallel assembly.
- `FEMX_ENABLE_PETSC=ON`: enable PETSc applications when PETSc and MPI are found.
- `FEMX_ENABLE_ENZYME=OFF`: enable Enzyme automatic differentiation support.
- `FEMX_BUILD_APPS=ON`: build applications.
- `FEMX_RESOLVE_BACKEND=AUTO`: requested ReSolve backend, one of `AUTO`, `CPU`,
  or `CUDA`.

If ReSolve is installed in a custom location, pass its install prefix:

```shell
$ cmake -S . -B build -DReSolve_ROOT=/path/to/resolve
```

To build with Enzyme, enable the option and pass Enzyme's CMake package
directory. The Enzyme package supplies the matching LLVM/Clang toolchain and
the `LLDEnzymeFlags` target used by femx:

```shell
$ cmake -S . -B build-enzyme \
    -DFEMX_ENABLE_ENZYME=ON \
    -DEnzyme_DIR=/path/to/Enzyme/enzyme/build
```

## Examples and applications

Run the inverse linear-control example when PETSc support is enabled:

```shell
$ ./build/examples/inverse/linear-control/example-inverse-linear-control
```

Run the forward Navier-Stokes GLS application:

```shell
$ ./build/apps/forward/navier-gls/navier-gls --config apps/forward/navier-gls/inputs/cavity/Config.json
```

Run the PETSc/MPI forward Navier-Stokes GLS application:

```shell
$ cmake -S . -B build-petsc \
    -DFEMX_ENABLE_PETSC=ON \
    -DFEMX_ENABLE_HDF5=ON \
    -DFEMX_ENABLE_RESOLVE=OFF
$ cmake --build build-petsc --target navier-gls-petsc
$ mpiexec -n 8 ./build-petsc/apps/forward/navier-gls-petsc/navier-gls-petsc \
    --config apps/forward/navier-gls-petsc/inputs/private/aneurysmA/Config.json \
    -ksp_rtol 1e-8 -ksp_max_it 5000
```

## Using femx in CMake

Inside this repository, link against the aggregate `femx::femx` target:

```cmake
add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE femx::femx)
```

Individual component targets are also available:

- `femx::core`: shared types and workspace choices.
- `femx::mesh`: mesh data structures and readers.
- `femx::fem`: finite element spaces, quadrature, and elements.
- `femx::linalg`: vectors, dense matrices, sparse matrices, and CSR patterns.
- `femx::system`: linear operators, system matrices/vectors, native and ReSolve solvers.
- `femx::system_petsc`: PETSc-backed system adapters, when PETSc is enabled.
- `femx::assembly`: FEM dof layouts, element kernels, assemblers, and sparsity builders.
- `femx::equation`: residual equations and state solvers.
- `femx::inverse`: objective, reduced-functional, and adjoint utilities.
- `femx::inverse_petsc`: PETSc TAO adapters, when PETSc is enabled.
- `femx::bc`, `femx::io`, and `femx::ad`: boundary conditions, output, and AD helpers.
