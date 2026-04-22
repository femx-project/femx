# refem

refem is a small C++ finite element library for experimenting with sparse linear solves.

The library currently provides basic mesh, finite element, assembly, boundary condition,
I/O, and linear algebra components. It also includes examples and applications,
including a diffusion example and an unsteady Navier-Stokes GLS solver.

## Getting started

Dependencies:

- CMake >= 3.22
- C++17 compiler
- ReSolve (optional, for the linear solver backend)
- HDF5 (optional, for HDF5/XDMF output)
- OpenMP (optional, for parallel assembly)

To build the library and applications:

```shell
$ cmake -S . -B build
$ cmake --build build
```

To build the diffusion example as well:

```shell
$ cmake -S . -B build -DREFEM_BUILD_EXAMPLES=ON
$ cmake --build build
```

## CMake options

Common configuration options are:

- `REFEM_ENABLE_RESOLVE=ON`: enable the ReSolve linear solver backend.
- `REFEM_ENABLE_HDF5=ON`: enable HDF5/XDMF output support.
- `REFEM_ENABLE_OPENMP=ON`: enable OpenMP parallel assembly.
- `REFEM_BUILD_APPS=ON`: build applications.

If ReSolve is installed in a custom location, pass its CMake package directory:

```shell
$ cmake -S . -B build -DReSolve_DIR=/path/to/resolve/lib/cmake/ReSolve
```

## Examples and applications

Run the diffusion example:

```shell
$ ./build/examples/diffusion/diffusion --backend cpu
```

Use `--backend cuda` to run through the CUDA workspace when ReSolve was built
with CUDA support.

Run the Navier-Stokes GLS application:

```shell
$ ./build/apps/navier-gls/navier-gls --config apps/navier-gls/inputs/cavity/Config.json
```

## Using refem in CMake

Inside this repository, link against the aggregate `refem::refem` target:

```cmake
add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE refem::refem)
```

Individual component targets are also available, such as `refem::mesh`,
`refem::fe`, `refem::forms`, `refem::linalg`, `refem::io`, and
`refem::solver`.

