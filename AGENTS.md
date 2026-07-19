# femx agent guidance

## C++ naming

- Keep public type and operation names descriptive. For local variables,
  function parameters, and private data members, prefer the established short
  forms: `jac`, `tr`, `sec`, `lin`, `init`, `ctx`, `hist`, `res`, `val`,
  `vals`, `src`, `ref`, `obs`, `obj`, `integ`, and `dst`.
- Use the same abbreviation consistently in declarations, definitions, and
  nearby tests. Prefer other conventional numerical abbreviations such as
  `mat`, `vec`, `prm`, `grad`, `adj`, `assm`, `sol`, `dir`, `dim`, `idx`, and
  `num` when their meaning is clear from the type and scope.
- Do not shorten prose, diagnostics, or a name when the abbreviation would be
  ambiguous outside its immediate numerical context.

