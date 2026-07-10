# Changelog

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
- ReSolve support currently targets the `develop` branch.
