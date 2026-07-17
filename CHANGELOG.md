# Changelog

## v0.2.0

This release adds the supported Python interface used by femx applications.

### Added

- The `femx` Python package with mesh, Navier--Stokes, observation, control,
  reduced-functional, optimization, and ensemble-basis APIs.
- Reusable Navier--Stokes model support in the native library.
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
