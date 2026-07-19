# Coding Conventions

This document describes the coding conventions used in femx.

The abbreviations below are representative examples, not a complete list of
allowed names. Other conventional abbreviations may be used when their meaning
is clear from the type and surrounding context.

## C++ naming

Keep public type and operation names descriptive.

For local variables, function parameters, and private data members, concise
names may be used when their meaning is clear.

| Short form | Full form                   |
| ---------- | --------------------------- |
| `jac`      | `jacobian`                  |
| `tr`       | `trajectory`                |
| `sec`      | `section`                   |
| `lin`      | `linear`, `linearization`   |
| `init`     | `initial`, `initialize`     |
| `ctx`      | `context`                   |
| `hist`     | `history`                   |
| `res`      | `residual`, `result`        |
| `val`      | `value`                     |
| `vals`     | `values`                    |
| `src`      | `source`                    |
| `dst`      | `destination`               |
| `ref`      | `reference`                 |
| `obs`      | `observation`               |
| `obj`      | `objective`                 |
| `integ`    | `integrator`, `integration` |
| `mat`      | `matrix`                    |
| `vec`      | `vector`                    |
| `prm`      | `parameter`, `parameters`   |
| `grad`     | `gradient`                  |
| `adj`      | `adjoint`                   |
| `assm`     | `assembly`, `assembler`     |
| `sol`      | `solution`                  |
| `dir`      | `direction`                 |
| `dim`      | `dimension`                 |
| `idx`      | `index`                     |
| `num`      | `number`                    |

Use the same abbreviation consistently in declarations, definitions, and
nearby tests.

Prefer the full name when an abbreviation would be ambiguous.

Do not shorten words in documentation, comments, diagnostics, or user-facing
messages.
