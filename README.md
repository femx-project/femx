# femx

femx is a C++ finite element library for forward and inverse analysis.

The v0.3.0 release adds a supported Python API for Navier--Stokes forward and
inverse workflows, including observations, controls, reduced functionals, and
optimization. The C++ and Python APIs remain pre-1.0 and may change between
minor releases.

## Requirements

- CMake >= 3.22
- C++ standard >= 17

Optional dependencies:

- HDF5, for HDF5/XDMF output
- Re::Solve built from the `develop` branch, for CPU/CUDA linear solver
  backends
- PETSc 3.19 or later (tested with PETSc 3.19.6)
- MPI, used with PETSc for linear solvers and TAO optimization
- OpenMP, for parallel assembly
- Enzyme + Clang, for automatic differentiation kernels
- Python >= 3.9, pybind11, NumPy, and SciPy, for the Python API

## Build

```shell
git clone --recursive https://github.com/femx-project/femx.git
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
cmake .. -DFEMX_ENABLE_RESOLVE=ON
make
```

For PETSc:

```shell
cmake .. -DFEMX_ENABLE_PETSC=ON -DPETSC_DIR=/path/to/petsc
make
```

Add `-DPETSC_ARCH=...` when using a PETSc build with an architecture directory.

## Python API

Build and install the supported Python package directly from the repository:

```shell
python3 -m pip install .
```

See [python/README.md](python/README.md) for details.

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
./examples/poisson-opt/poisson-opt-resolve --nx 32 --ny 32 --output yes --max-its 50
```

See [examples/poisson-opt](examples/poisson-opt) for the optimization problem
definition.

## Run Apps

The Navier-Stokes forward app provides separate configuration sets for ReSolve
and PETSc. From your build directory, run the executable for the backend you
enabled:

```shell
./apps/ns-forward/ns-forward-resolve \
  --config ../apps/ns-forward/configs/resolve/cavity/Config.json

./apps/ns-forward/ns-forward-petsc \
  --config ../apps/ns-forward/configs/petsc/cavity/Config.json
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
- `FEMX_BUILD_PYTHON=ON|OFF`

## Using femx in CMake

Inside this repository, link against the aggregate target:

```cmake
add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE femx::femx)
```

Component targets such as `femx::linalg`, `femx::ad`, `femx::fem`,
`femx::assembly`, `femx::state`, `femx::inverse`, and `femx::io` are also
available. `femx::opt` is available when PETSc is enabled.

`femx::ad` exposes the optional C++ Enzyme entry point in
`<femx/ad/Enzyme.hpp>`. Enable it with a Clang compiler and
`FEMX_ENABLE_ENZYME=ON`; the Python API does not expose Enzyme yet.

## Documentation

```shell
./preview-docs.sh
```

## License

femx is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE).
