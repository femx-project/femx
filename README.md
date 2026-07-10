# femx

femx is a small C++ finite element library for forward and inverse analysis.

The v0.1.0 release focuses primarily on forward workflows. Inverse-problem and optimization utilities are still experimental, and the current optimization examples use PETSc/TAO.

## Example Result

![Poisson optimization result](docs/figs/poisson-opt.png)

## Requirements

- CMake >= 3.22
- C++17 compiler

Optional dependencies:

- HDF5, for HDF5/XDMF output
- ReSolve, for ReSolve linear solver backends. The current backend expects the
  ReSolve `develop` branch; released ReSolve versions may not work.
- PETSc and MPI, for PETSc linear solvers and TAO optimization
- OpenMP, for parallel assembly
- Enzyme + Clang, for automatic differentiation kernels

## Build

```shell
cmake -S . -B build
cmake --build build
```

Enable optional dependencies explicitly. If an enabled dependency is not found,
CMake fails during configuration.

For HDF5 output:

```shell
cmake -S . -B build -DFEMX_ENABLE_HDF5=ON
cmake --build build
```

For ReSolve:

```shell
cmake -S . -B build -DFEMX_ENABLE_RESOLVE=ON -DReSolve_ROOT=/path/to/resolve
```

For PETSc:

```shell
cmake -S . -B build -DFEMX_ENABLE_PETSC=ON -DPETSC_DIR=/path/to/petsc
```

Add `-DPETSC_ARCH=...` when using a PETSc build with an architecture directory.

## Examples and Apps

Examples and applications are backend-specific:

- `*-resolve` targets use ReSolve.
- `*-petsc` targets use PETSc and MPI.
- Optimization examples use PETSc/TAO, even when the linear solves use ReSolve.
- A build without enabled solver backends does not provide runnable backend
  examples.

## CMake Options

Common options:

- `FEMX_ENABLE_HDF5=ON|OFF`
- `FEMX_ENABLE_RESOLVE=ON|OFF`
- `FEMX_RESOLVE_BACKEND=AUTO|CPU|CUDA`
- `FEMX_ENABLE_PETSC=ON|OFF`
- `PETSC_DIR=/path/to/petsc`
- `PETSC_ARCH=...`
- `FEMX_ENABLE_OPENMP=ON|OFF`
- `FEMX_ENABLE_ENZYME=ON|OFF`
- `FEMX_BUILD_EXAMPLES=ON|OFF`
- `FEMX_BUILD_APPS=ON|OFF`
- `FEMX_BUILD_TESTS=ON|OFF`

## Using femx in CMake

Inside this repository, link against the aggregate target:

```cmake
add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE femx::femx)
```

Component targets such as `femx::linalg`, `femx::fem`, `femx::assembly`,
`femx::state`, `femx::inverse`, and `femx::io` are also available. `femx::opt`
is available when PETSc is enabled.

## Documentation

```shell
./preview-docs.sh
```

## License

femx is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE).
