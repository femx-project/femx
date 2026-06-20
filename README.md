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
- `FEMX_OPENMP_ROOT=/path/to/libomp-prefix`: optional LLVM OpenMP runtime
  prefix for Clang builds.
- `FEMX_ENABLE_PETSC=ON`: enable PETSc applications when PETSc and MPI are found.
- `FEMX_ENABLE_ENZYME=OFF`: enable Enzyme automatic differentiation support
  when set to `ON`.
- `FEMX_BUILD_APPS=ON`: build applications.
- `FEMX_RESOLVE_BACKEND=AUTO`: requested ReSolve backend, one of `AUTO`, `CPU`,
  or `CUDA`.

If ReSolve is installed in a custom location, pass its install prefix:

```shell
$ cmake -S . -B build -DReSolve_ROOT=/path/to/resolve
```

To build with Enzyme, enable the option. If Enzyme installs a CMake package,
pass that directory. Otherwise femx will also look for `ClangEnzyme` or
`LLVMEnzyme` plugin libraries through `Enzyme_ROOT`/`ENZYME_ROOT`, or through a
direct plugin path:

```shell
$ cmake -S . -B build-enzyme \
    -DFEMX_ENABLE_ENZYME=ON \
    -DEnzyme_DIR=/path/to/Enzyme/enzyme/build
```

```shell
$ cmake -S . -B build-enzyme \
    -DFEMX_ENABLE_ENZYME=ON \
    -DEnzyme_CLANG_PLUGIN=/path/to/ClangEnzyme-18.so
```

For a PETSc + Enzyme build, use the preset. It assumes Enzyme is installed
under `$HOME/opt/enzyme-llvm18`:

```shell
$ cmake --preset petsc-enzyme
$ cmake --build --preset petsc-enzyme --target make-obs ns-var-petsc
```

If Enzyme is installed elsewhere, pass either the Enzyme CMake package
directory or the plugin path when configuring:

```shell
$ cmake --preset petsc-enzyme \
    -DEnzyme_DIR=/path/to/enzyme/lib/cmake/Enzyme
```

```shell
$ cmake --preset petsc-enzyme \
    -DEnzyme_CLANG_PLUGIN=/path/to/ClangEnzyme-18.so
```

Enzyme builds use Clang, so OpenMP requires LLVM `libomp`. If CMake cannot find
OpenMP automatically, install `libomp` or pass its prefix:

```shell
$ cmake -S . -B build-enzyme \
    -DFEMX_ENABLE_ENZYME=ON \
    -DEnzyme_DIR=/path/to/enzyme/lib/cmake/Enzyme \
    -DFEMX_OPENMP_ROOT=/path/to/libomp-prefix
```

## Examples and applications

Run the reduced-functional linear-control example using the new `problem` and
`solve` APIs:

```shell
$ ./build/examples/inverse/linear-control-new-api/example-inverse-linear-control-new-api
```

Run the PETSc/TAO inverse linear-control example when PETSc support is enabled:

```shell
$ ./build/examples/inverse/linear-control/example-inverse-linear-control
```

Run the forward Navier-Stokes GLS application:

```shell
$ ./build/apps/forward/ns-gls/ns-gls --config apps/forward/ns-gls/inputs/cavity/Config.json
```

Run the PETSc/MPI forward Navier-Stokes GLS application:

```shell
$ cmake -S . -B build-petsc \
    -DFEMX_ENABLE_PETSC=ON \
    -DFEMX_ENABLE_HDF5=ON \
    -DFEMX_ENABLE_RESOLVE=OFF
$ cmake --build build-petsc --target ns-gls-petsc
$ mpiexec -n 8 ./build-petsc/apps/forward/ns-gls-petsc/ns-gls-petsc \
    --config apps/forward/ns-gls-petsc/inputs/private/aneurysmA/Config.json \
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
- `femx::algebra`: vectors, matrices, linear operators, and solver interfaces.
- `femx::fem`: mesh-facing FEM data structures, spaces, quadrature, and elements.
- `femx::assembly`: FEM kernels, assemblers, sparsity builders, and residual adapters.
- `femx::problem`: residual, objective, observation, and time-residual interfaces.
- `femx::solve`: Newton, time-stepping, and reduced-functional utilities.
- `femx::optimize`: PETSc/TAO optimization adapters, when PETSc is enabled.
- `femx::io`: output and data readers.

Preferred include paths use the new public component layout:

```cpp
#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/solve/Newton.hpp>
```

The pre-refactor component targets and include paths have been removed. Update
old code to the public component layout above before linking against femx.
