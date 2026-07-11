# femx

femx is a small C++ finite element library for forward and inverse analysis.

The v0.1.0 release focuses primarily on forward workflows. Inverse-problem and optimization utilities are still experimental, and the current optimization examples use PETSc/TAO.

## Requirements

- CMake >= 3.22
- C++17 compiler

Optional dependencies:

- HDF5, for HDF5/XDMF output
- Re::Solve 0.99.2, for CPU/GPU linear solver backends.
- PETSc and MPI, for linear solvers and TAO optimization
- OpenMP, for parallel assembly
- Enzyme + Clang, for automatic differentiation kernels

## Build

```shell
git clone --recursive https://github.com/kakeueda/femx.git
cd femx
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/femx
make
```

If you already cloned femx without submodules, initialize them first:

```shell
git submodule update --init --recursive
```

Enable optional dependencies explicitly. If an enabled dependency is not found,
CMake fails during configuration. Add these flags to the `cmake ..` command
above when needed.

For HDF5 output:

```shell
cmake .. -DFEMX_ENABLE_HDF5=ON
make
```

For ReSolve:

```shell
cmake .. -DFEMX_ENABLE_RESOLVE=ON -DReSolve_ROOT=/path/to/resolve
make
```

For PETSc:

```shell
cmake .. -DFEMX_ENABLE_PETSC=ON -DPETSC_DIR=/path/to/petsc
make
```

Add `-DPETSC_ARCH=...` when using a PETSc build with an architecture directory.

## Install

```shell
make install
```

Run the install test before packaging a release:

```shell
make test_install
```

## Run Examples

The default Poisson example uses the native dense solver and does not require
optional solver packages. Run it from your build directory:

```shell
./examples/poisson/poisson --output yes
```

See [examples/poisson](examples/poisson) for the problem definition and
available solver variants.

Backend-specific examples are available when their dependencies are enabled:

```shell
./examples/poisson/poisson-resolve --nx 32 --ny 32 -b cpu --output yes
```

Optimization examples use PETSc/TAO, even when the linear solves use ReSolve:

```shell
./examples/poisson-opt/poisson-opt-resolve --nx 32 --ny 32 -b cpu --output yes --max-its 50
```

See [examples/poisson-opt](examples/poisson-opt) for the optimization problem
definition.

## Run Apps

The Navier-Stokes forward app provides separate configuration sets for ReSolve
and PETSc. From your build directory, run the executable for the backend you
enabled:

```shell
./apps/ns-forward/ns-forward-resolve \
  --config ../../apps/ns-forward/configs/resolve/cavity/Config.json

./apps/ns-forward/ns-forward-petsc \
  --config ../../apps/ns-forward/configs/petsc/cavity/Config.json
```

See [apps/ns-forward](apps/ns-forward) for the formulation and available demo
configurations.

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
