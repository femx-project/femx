API Reference
=============

The Doxygen API reference is generated into ``docs/doxygen/html``. Generated
HTML is not tracked in git.

Open ``html/index.html`` after running:

.. code-block:: bash

   doxygen docs/doxygen/Doxyfile.in

On remote or minimal Linux systems without a desktop browser, serve the
generated HTML from the repository root:

.. code-block:: bash

   python3 -m http.server 8000 --directory docs/doxygen/html

Then open ``http://localhost:8000/`` in a browser.

For a one-command local preview:

.. code-block:: bash

   ./docs/preview-doxygen.sh
