****
femx
****

femx is a research finite-element code for forward and inverse PDE workflows.
The v0.1.0 documentation is intentionally small: it introduces the project,
points users at the examples, and links to Doxygen-generated API references.

Documentation
-------------

Start with the Poisson examples to see the current command-line interface and
VTU visualization output. Source code documentation generated with Doxygen is
available under ``doxygen/html/index.html`` after building the docs.

Building The Docs
-----------------

From the repository root:

.. code-block:: bash

   doxygen docs/doxygen/Doxyfile.in

If Sphinx is installed, the landing page can also be built with:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Examples
--------

The forward Poisson examples accept mesh dimensions, solver backend, and VTU
output selection:

.. code-block:: bash

   poisson-resolve --nx 48 --ny 48 -b cpu --output yes

The optimization example follows the same shape and also exposes optimization
parameters:

.. code-block:: bash

   poisson-opt-resolve --nx 48 --ny 48 -b cpu --output yes --max-its 50

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   doxygen/index
