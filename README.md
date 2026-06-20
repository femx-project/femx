# femx

femx is a small C++ finite element library for experimenting with sparse linear solves.

The public library is organized around core types, algebra, FEM data,
assembly, generic problem interfaces, solvers, optional optimizer adapters,
and I/O.

## Getting started

Dependencies:

- CMake >= 3.22
- C++17 compiler
- ReSolve (optional, for the linear solver backend)
- Enzyme + Clang (optional, for automatic differentiation kernels)
- HDF5 (optional, for HDF5/XDMF output)
- OpenMP (optional, for parallel assembly)
- PETSc and MPI (optional, for PETSc backends and TAO optimization)

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
- `FEMX_ENABLE_ENZYME=OFF`: enable Enzyme automatic differentiation support.
  This requires a Clang toolchain and an Enzyme plugin or CMake package.
- `FEMX_OPENMP_ROOT=/path/to/libomp-prefix`: optional LLVM OpenMP runtime
  prefix for Clang builds.
- `FEMX_ENABLE_PETSC=ON`: enable PETSc backends and optimizer adapters when
  PETSc and MPI are found.
- `FEMX_BUILD_APPS=ON`: build applications.
- `FEMX_RESOLVE_BACKEND=AUTO`: requested ReSolve backend, one of `AUTO`, `CPU`,
  or `CUDA`.

If ReSolve is installed in a custom location, pass its install prefix:

```shell
$ cmake -S . -B build -DReSolve_ROOT=/path/to/resolve
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
- `femx::ad`: Enzyme declarations and small AD helpers.
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
