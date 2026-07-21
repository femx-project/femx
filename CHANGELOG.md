# Changelog

## v0.4.0

v0.4.0 refactors the linear algebra API by separating storage from
backend-specific operations and adds explicit CSR transpose support for Host
and CUDA backends.

### Added

- Host and CUDA `VectorHandler` and `MatrixHandler` APIs for vector operations.
- CSR transpose support on Host and CUDA, with cached structure and value 
  mappings for efficient repeated updates.
- Memory-space-to-backend mappings, stronger view validation, and expanded
  Host and CUDA linear algebra test coverage.

### Changed

- Renamed `CsrGraph` to `CsrPattern` and reorganized dense and CSR matrices,
  vectors, and linear solvers around separate storage and backend-specific
  handler interfaces.
- Centralized shared cuBLAS and cuSPARSE handle management and migrated
  assembly, state, inverse, Navier-Stokes, examples, and Python bindings to
  the revised linear algebra API.

## v0.3.0

This is a major pre-1.0 architectural update that introduces built-in CPU
and CUDA execution backends across femx's core architecture.

### Added

- Built-in Host and CUDA backends, including device vectors, CSR storage,
  finite-element assembly, boundary handling, and ReSolve linear solves.
- End-to-end CUDA execution for the Poisson example and Navier-Stokes forward
  application.
- CUDA-compatible Enzyme automatic differentiation for Navier-Stokes
  Jacobian products, enabling device-side adjoint and reduced-gradient
  workflows in Clang CUDA builds.
- New geometry, assembly-map, boundary-map, and control-map abstractions, with
  expanded CPU, CUDA, PETSc, and ReSolve test coverage.

### Changed

- Redesigned linear algebra, assembly, state, time-integration, inverse, and
  Navier-Stokes APIs around backend-aware interfaces and explicit execution
  contexts. The examples and Python bindings now use the same architecture.
- Unified the Navier-Stokes row operator across PETSc Host and ReSolve CUDA
  assembly paths.

## v0.2.0

This release adds the supported Python interface used by femx applications.

### Added

- The `femx` Python package with mesh, Navier-Stokes, observation, control,
  reduced-functional, optimization, and ensemble-basis APIs.
- Reusable Navier-Stokes model support in the native library.
- Boundary-surface utilities, time-dependent Dirichlet data, and block time
  regularization.
- Python API tests and continuous-integration coverage.

### Changed

- Replaced the `femx-experimental` distribution with the supported `femx`
  Python distribution.
- Expanded and aligned the state, inverse, and time-dependent APIs used by the
  Python bindings.
- Updated the ReSolve adapter for the current ReSolve development API.

### Notes

- The default Python build enables PETSc and HDF5.
- The public APIs remain pre-1.0 and may change between minor releases.

## v0.1.0

Initial public release of femx.

### Added

- Core finite-element mesh, space, assembly, and linear algebra utilities.
- Forward Poisson examples with native dense, ReSolve, and PETSc solver paths.
- Experimental Poisson boundary-control optimization example using PETSc/TAO.
- Navier-Stokes forward application support.
- VTU and HDF5/XDMF visualization outputs.

### Notes

- Inverse-problem and optimization APIs are experimental.
- ReSolve support targets the ReSolve 0.99.2 release.
