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

## Declaration spacing

Place a blank line between class declarations and between function
declarations. Do not add blank lines between consecutive type aliases.

```cpp
using Index = int;
using Value = double;

void initialize();

void finalize();
```

## Doxygen documentation

Document public classes and functions at their declarations in header files.
Private APIs do not require Doxygen comments. Use `/** ... */` blocks and write
`@brief` as a short imperative sentence, such as `Compute ...`, `Return ...`,
or `Set ...`.

Every public function, including getters and setters, requires documentation.
Documentation may be omitted for trivial default constructors, destructors,
and copy or move operations.

When a function or class has no documentation tags and a brief description is
sufficient, write the entire comment on one line:

```cpp
/** @brief Clear all cached data. */
void clear();
```

Use the following tags whenever the corresponding element is present:

- `@param` for every function parameter;
- `@return` for every non-`void` return value;
- `@throws` for each exception that is important to the public contract.

Document accessors with only a one-line `@brief`.

Separate `@brief` or a detailed description from the first tag with one blank
comment line. Do not add blank lines between tags.

```cpp
/**
 * @brief Return the value at an index.
 *
 * @param[in] index - Value index.
 * @return Value at `index`.
 * @throws std::runtime_error - If `index` is out of range.
 */
Real value(Index index) const;
```

Use the form `@param[direction] name - Description.`, where the optional
direction is:

- `[in]` for values read without being modified;
- `[out]` for values replaced by the function;
- `[in,out]` for values both read and modified.

Order tags as `@param`, `@return`, `@throws`, `@note`, and `@warning`. Do not add
`@throws` to `noexcept` functions or list incidental allocation failures unless
they are part of the public contract.

Document each class member variable with a trailing `///<` comment:

```cpp
int rows_; ///< Number of rows.
```
