---
name: python-api
description: Coordinate changes to the supported femx Python API across C++ declarations, pybind11 bindings, Python wrappers, public exports, tests, and documentation. Use when adding, changing, or removing Python-visible types, functions, options, callbacks, arrays, or behavior under `python/bindings` or `python/femx`, or when a C++ change affects the Python API.
---

# Update the femx Python API

1. Determine whether the change targets the supported API or
   `python/experimental`. Keep the two APIs separate unless the user explicitly
   requests both.
2. Trace the affected public surface through the relevant layers:
   - C++ declarations and behavior under `femx/`
   - pybind11 definitions under `python/bindings/`
   - Python wrappers under `python/femx/`
   - package exports in `python/femx/__init__.py`
   - focused tests under `python/tests/`
   - user documentation in `python/README.md` or nearby documentation
3. Update only the layers required by the public behavior. Keep names,
   defaults, accepted values, return shapes, and error behavior consistent
   across the binding and wrapper boundary.
4. Preserve object lifetime, callback exception handling, NumPy ownership,
   mutability, shape, and contiguity semantics when they are relevant. Add a
   focused regression test for any changed boundary behavior.
5. Keep `__all__` and package-level imports synchronized when adding, removing,
   or renaming public Python symbols.
6. Follow `AGENTS.md`, `docs/CONVENTIONS.md`, and nearby binding and test
   patterns. Avoid unrelated API cleanup.
7. Apply the `verify` skill after implementation. Start with the directly
   affected Python test module and do not configure additional backend variants
   solely for completeness.
8. Report the layers changed, the focused checks run, and relevant optional
   backends or Python configurations not exercised.
