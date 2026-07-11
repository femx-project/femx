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

The forward Poisson examples accept mesh dimensions, solver backend, and VTU
output selection. The ``poisson`` target uses the native dense solver for
small dependency-free checks:

.. code-block:: bash

   poisson --output yes

Optional solver backends are exposed through backend-specific targets:

.. code-block:: bash

   poisson-resolve --nx 32 --ny 32 -b cpu --output yes

The optimization example follows the same shape and also exposes optimization
parameters:

.. code-block:: bash

   poisson-opt-resolve --nx 32 --ny 32 -b cpu --output yes --max-its 50

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   doxygen/index
