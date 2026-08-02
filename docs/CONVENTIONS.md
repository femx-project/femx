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
| `ctr`      | `control`                   |
| `traj`     | `trajectory`                |
| `trans`    | `transpose`                 |
| `lin`      | `linear`, `linearization`   |
| `init`     | `initial`, `initialize`     |
| `ctx`      | `context`                   |
| `hist`     | `history`                   |
| `res`      | `residual`                  |
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
| `dir`      | `direction`                 |
| `dim`      | `dimension`                 |
| `idx`      | `index`                     |
| `num`      | `number`                    |
| `itrs`     | `iterations`                |
| `opts`     | `options`                   |
| `config`   | `configuration`             |

When a local variable, function parameter, or private data member needs a
memory-space qualifier, prefix it with `h_` for Host storage and `d_` for
Device storage. This is especially useful when the same logical data exists
in both memory spaces:

```cpp
HostVector<Real> h_state;
DeviceVector<Real> d_state;
```
Do not shorten words in documentation, comments, diagnostics, or user-facing
messages.

## Loop indices

Use the following names when traversing matrix storage:

| Index | Iteration domain          |
| ----- | ------------------------- |
| `i`   | matrix row                |
| `j`   | matrix column             |
| `k`   | stored matrix entry       |

Use the following semantic names for finite-element loop indices:

| Index | Iteration domain    |
| ----- | ------------------- |
| `in`  | node                |
| `ic`  | component           |
| `id`  | spatial dimension   |
| `ie`  | element             |

## Doxygen documentation

Document public classes and functions at their declarations in header files.
Private APIs do not require Doxygen comments. Use `/** ... */` blocks and write
`@brief` as a short imperative sentence, such as `Compute ...`, `Return ...`,
or `Set ...`.

Every public function, including getters and setters, requires documentation.
Documentation may be omitted for trivial default constructors, destructors,
and copy or move operations.

Use the following tags whenever the corresponding element is present:

- `@param` for every function parameter;
- `@return` for every non-`void` return value;
- `@throws` for each exception that is important to the public contract.

Document accessors with only an `@brief`. Use a multi-line `/** ... */` block
even when the documentation fits on one line.


```cpp
/**
 * @brief Return the value at an index.
 *
 * @param[in] index - Value index.
 * @return Value at `index`.
 * @throws std::runtime_error If validation fails.
 */
Real value(Index index) const;
```

Within each documentation block, align parameter names after the longest
direction marker used by that block. Do not add padding when all direction
markers have the same width. For example:

```cpp
 * @param[in]     src - Source values.
 * @param[in,out] dst - Destination values.
```

The directions are:

- `[in]` for values read without being modified;
- `[out]` for values replaced by the function;
- `[in,out]` for values both read and modified.

Use the form `@throws std::runtime_error If validation fails.`.

Document each class member variable with a trailing `///<` comment:

```cpp
int rows_; ///< Number of rows.
```
