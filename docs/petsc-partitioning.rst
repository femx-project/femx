****************************
PETSc MPI Graph Partitioning
****************************

femx partitions every square Host CSR layout before its first PETSc matrix
assembly. ``MatPartitioning`` uses ParMETIS by default when the selected PETSc
provides it. PT-Scotch and PETSc's average partitioner are the respective
fallbacks.

The application degree-of-freedom numbering remains unchanged. femx converts
it to a rank-major PETSc numbering, gives each PETSc rank the corresponding
matrix rows and vector entries, and assigns each finite element to the rank
that owns most of its residual rows. The partition is cached with the CSR
layout and reused by later nonlinear or time steps.

Build PETSc with ParMETIS
========================

The following builds an optimized PETSc 3.19.6 in place. PETSc downloads
matching METIS and ParMETIS sources, so no system-specific library paths are
needed:

.. code-block:: bash

   git clone --branch v3.19.6 --depth 1 \
     https://gitlab.com/petsc/petsc.git petsc-3.19.6
   cd petsc-3.19.6
   ./configure \
     PETSC_ARCH=linux-parmetis-opt \
     --with-debugging=0 \
     --with-shared-libraries=1 \
     --with-cc=mpicc \
     --with-cxx=mpicxx \
     --with-fc=0 \
     --with-cuda=0 \
     --download-metis \
     --download-parmetis \
     COPTFLAGS=-O3 \
     CXXOPTFLAGS=-O3
   make PETSC_DIR="$PWD" PETSC_ARCH=linux-parmetis-opt all
   make PETSC_DIR="$PWD" PETSC_ARCH=linux-parmetis-opt check

Existing METIS and ParMETIS installations can instead be selected with
PETSc's ``--with-metis-*`` and ``--with-parmetis-*`` configure options. Their
integer widths must match PETSc.

Select the build in femx
========================

Configure femx with the matching PETSc source and architecture directories.
The requirement option prevents pkg-config from silently selecting another
PETSc installation:

.. code-block:: bash

   cmake -S . -B build-parmetis \
     -DCMAKE_BUILD_TYPE=Release \
     -DFEMX_ENABLE_PETSC=ON \
     -DFEMX_REQUIRE_PETSC_PARMETIS=ON \
     -DPETSC_DIR=/path/to/petsc-3.19.6 \
     -DPETSC_ARCH=linux-parmetis-opt
   cmake --build build-parmetis -j
   ctest --test-dir build-parmetis -R PETScMpiLinalgTests \
     --output-on-failure

CMake reports ``PETSc graph partitioner: ParMETIS (default)`` when the intended
configuration is selected.

Runtime options
===============

The partitioner uses the ``femx_`` PETSc options prefix. For example:

.. code-block:: bash

   mpiexec -n 4 build-parmetis/examples/poisson/poisson-petsc \
     --nx 256 --ny 256 --output no \
     -femx_mat_partitioning_view

Use ``-femx_mat_partitioning_type ptscotch`` or
``-femx_mat_partitioning_type average`` to compare another PETSc partitioner.

Current distribution boundary
=============================

Matrix rows, KSP vectors, and element work use the graph partition. Public
Host state and residual vectors are still replicated on every MPI rank; femx
therefore still performs a residual ``MPI_Allreduce`` and gathers solved PETSc
vectors. A fully distributed state/ghost-vector design is a separate step and
is required to remove those remaining global communications.
