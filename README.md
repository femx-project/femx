# femx

femx is a C++ finite element library for forward and inverse analysis.

Current features include:

- Finite-element utilities for meshes, function spaces, quadrature, and sparse assembly.
- Inverse-analysis components for controls, observations, objectives, regularization, and adjoint gradients.
- A Navier–Stokes forward application and Poisson forward and inverse examples.
- Native CPU, ReSolve CPU/CUDA, and PETSc MPI execution paths.
- Python bindings and HDF5/XDMF output.

Note: The C++ and Python APIs are pre-1.0 and may change between minor releases.

## Requirements

- CMake >= 3.22
- C++ standard >= 17

Optional dependencies:

- HDF5, for HDF5/XDMF output
- Re::Solve built from the `develop` branch, for Host/Device linear solves
- PETSc 3.19 or later (tested with PETSc 3.19.6); a ParMETIS-enabled build is
  recommended for MPI graph partitioning
- MPI, used with PETSc for linear solvers and TAO optimization
- OpenMP, for parallel assembly
- Enzyme + Clang, for automatic differentiation kernels; the CUDA path is
  experimental
- Python >= 3.9 with pip, pybind11, NumPy, and SciPy, for the Python API

The CUDA Enzyme path is experimental. Validate its derivatives against finite
differences for the intended Clang/Enzyme/CUDA/GPU combination before relying
on them. Builds without Enzyme use analytic derivative paths where available.

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
Add `-DFEMX_REQUIRE_PETSC_PARMETIS=ON` to reject a PETSc build that does not
provide ParMETIS.

## Python API

From the repository root, activate a Python environment that provides pip and
install the supported Python package:

```shell
python3 -m pip install .
```

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

Optional solver examples are available when their dependencies are enabled:

```shell
./examples/poisson/poisson-resolve --nx 32 --ny 32 -b cpu --output yes
```

Optimization examples use PETSc/TAO, even when the linear solves use ReSolve:

```shell
./examples/poisson-opt/poisson-opt-resolve \
  -b cpu --nx 32 --ny 32 --output yes --max-its 50

./examples/poisson-opt/poisson-opt-resolve \
  -b cuda --nx 32 --ny 32 --output yes --max-its 50

mpiexec -n 4 ./examples/poisson-opt/poisson-opt-petsc \
  --nx 32 --ny 32 --output yes --max-its 50
```

See [examples/poisson-opt](examples/poisson-opt) for the optimization problem
definition, Enzyme configuration, and backend details.

## Run Apps

The Navier-Stokes forward app provides separate configuration sets for ReSolve
and PETSc. From the top-level `build/` directory created in the Build section,
run the executable for the solver you enabled:

```shell
./apps/navier/navier-resolve \
  --config ../apps/navier/configs/resolve/cavity/Config.json

./apps/navier/navier-petsc \
  --config ../apps/navier/configs/petsc/cavity/Config.json
```

See [apps/navier](apps/navier) for the formulation and available demo
configurations.

## CMake Options

Common options:

- `FEMX_ENABLE_HDF5=ON|OFF`
- `FEMX_ENABLE_RESOLVE=ON|OFF`
- `FEMX_RESOLVE_DEVICE=AUTO|HOST|DEVICE`
- `FEMX_ENABLE_PETSC=ON|OFF`
- `FEMX_REQUIRE_PETSC_PARMETIS=ON|OFF`
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

`femx::ad` exposes the C++ Enzyme entry point in `<femx/ad/Enzyme.hpp>`.
Enable it with a Clang compiler and `FEMX_ENABLE_ENZYME=ON`; the Python API
does not expose Enzyme yet.

## Documentation

```shell
./preview-docs.sh
```

## License

femx is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE).
