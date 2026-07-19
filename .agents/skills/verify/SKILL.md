---
name: verify
description: Select and run focused builds and tests for femx changes. Use after modifying C++, CUDA, CMake, Python bindings, applications, examples, or tests when verification is requested or needed before completing the task.
---

# Verify femx changes

1. Inspect the changed files and identify the affected component and backend.
2. Reuse an existing configured build directory whenever possible.
3. Build the narrowest relevant target and run the smallest directly related
   test first.
4. Expand verification only when the focused check fails to cover a material
   risk introduced by the change.
5. Do not configure or test additional CUDA, PETSc, ReSolve, Enzyme, or Python
   variants solely for completeness.
6. Report the commands run, their results, and any relevant checks not run.

Follow the repository `AGENTS.md`, `docs/CONVENTIONS.md`, and nearby test
patterns. Preserve unrelated working-tree changes.
