****
femx
****

femx is a research finite-element code for forward and inverse PDE workflows.
v0.9.0 streamlines the public finite-element and linear-algebra APIs around
``Mesh``, ``ElementShape``, and memory-space-specific vector and matrix
handlers.

Documentation
-------------

Start with the Poisson examples to see the current command-line interface and
VTU visualization output. Source code documentation generated with Doxygen is
generated under ``doxygen/html/index.html`` when building the docs. Generated
HTML is not tracked in git.

Building The Docs
-----------------

From the repository root:

.. code-block:: bash

   doxygen docs/doxygen/Doxyfile.in

Open ``docs/doxygen/html/index.html`` in a browser. On remote or minimal Linux
systems without a desktop browser, serve the generated HTML instead:

.. code-block:: bash

   python3 -m http.server 8000 --directory docs/doxygen/html

Then open ``http://localhost:8000/`` in a browser.

For a one-command local preview:

.. code-block:: bash

   ./preview-docs.sh

If Sphinx is installed, the landing page can also be built with:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Examples
--------

The forward Poisson examples accept mesh dimensions, execution backend, and VTU
output selection. The ``poisson`` target uses the Host dense linear system
for small dependency-free checks:

.. code-block:: bash

   poisson --output yes

Optional linear-system implementations are exposed through solver-specific
targets:

.. code-block:: bash

   poisson-resolve --nx 32 --ny 32 -b cpu --output yes

The optimization example follows the same shape and also exposes optimization
parameters:

.. code-block:: bash

   poisson-opt-resolve -b cpu --nx 32 --ny 32 --max-its 50
   poisson-opt-resolve -b cuda --nx 32 --ny 32 --max-its 50
   mpiexec -n 4 poisson-opt-petsc --nx 32 --ny 32 --max-its 50

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   doxygen/index
