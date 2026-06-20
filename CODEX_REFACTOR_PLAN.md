# femx Refactor Plan for Codex

This file is an implementation guide for incrementally moving `femx/` toward a simpler architecture. Treat it as a task list for Codex. The goal is to reduce API surface, clarify dependency direction, and avoid the current proliferation of `Matrix...`, `Operator...`, `Time...`, `system`, `eq`, and `inverse` concepts.

## 0. High-level objective

Refactor the library toward this public structure:

```text
femx/
  core/
    Types.hpp
    Workspace.hpp

  algebra/
    Vector.hpp
    DenseMatrix.hpp
    SparseMatrix.hpp
    LinearOperator.hpp
    MatrixBuilder.hpp
    LinearSolver.hpp
    backends/
      native/
      petsc/
      resolve/

  fem/
    Mesh.hpp
    Cell.hpp
    FESpace.hpp
    MixedFESpace.hpp
    FiniteElement.hpp
    Quadrature.hpp
    DofMap.hpp
    DofLayout.hpp
    BoundaryCondition.hpp

  assembly/
    Kernel.hpp
    Assembler.hpp
    SparsityPatternBuilder.hpp
    FEMResidual.hpp

  problem/
    Residual.hpp
    TimeResidual.hpp
    Objective.hpp
    Observation.hpp
    Linearization.hpp

  solve/
    Newton.hpp
    TimeStepper.hpp
    ReducedFunctional.hpp
    DerivativeCheck.hpp

  optimize/
    TaoOptimizer.hpp

  io/
    DataOut.hpp
    XdmfWriter.hpp
    Hdf5Writer.hpp
```

## 1. Non-negotiable constraints

1. Preserve numerical behavior. Refactoring must not intentionally change solver results, residual signs, Jacobian conventions, adjoint signs, or time-stepping semantics.
2. Keep the aggregate CMake target `femx::femx` working throughout the migration.
3. Keep old include paths working during the transition by adding forwarding headers, not by deleting old headers immediately.
4. Prefer small, buildable steps. After each phase, the project should configure and build with default CMake options as far as dependencies allow.
5. Do not rewrite examples and applications in the same commit as low-level file moves unless required by compilation.
6. Do not introduce new external dependencies.
7. Do not expose backend-specific types through generic problem or solve interfaces unless absolutely required.
8. Use `git mv` for file moves when working in a repository checkout.
9. Keep PETSc and ReSolve optional.
10. Keep C++17 compatibility.

## 2. Target dependency direction

The desired dependency direction is:

```text
core
  ↓
algebra
  ↓
problem
  ↓
solve
  ↓
optimize adapters

fem
  ↓
assembly
  ↓
problem implementations

io may depend on core, algebra, mesh/fem data types, but not solve/optimize.
backends may depend on algebra, core, and external libraries only.
```

Important rule: generic `problem` and `solve` code must not depend on FEM boundary-condition classes. FEM/BC-specific concepts should live in `fem`, `assembly`, or FEM-specific problem implementations.

## 3. Current-to-target mapping

Use this table as the migration map.

| Current area | Target area | Notes |
|---|---|---|
| `femx/common/Types.hpp` | `femx/core/Types.hpp` | Add forwarding header at old path. |
| `femx/common/Workspace.hpp` | `femx/core/Workspace.hpp` | Add forwarding header at old path. |
| `femx/linalg/Vector.hpp` | `femx/algebra/Vector.hpp` | Keep namespace `femx`; do not rename class yet. |
| `femx/linalg/DenseMatrix.hpp` | `femx/algebra/DenseMatrix.hpp` | Keep old include path as forwarding header. |
| `femx/linalg/SparseMatrix.hpp` | `femx/algebra/SparseMatrix.hpp` | Keep old include path as forwarding header. |
| `femx/system/LinearOperator.hpp` | `femx/algebra/LinearOperator.hpp` | Prefer namespace `femx::algebra` later; first phase may keep `femx::system` aliases. |
| `femx/system/LinearSolver.hpp` | `femx/algebra/LinearSolver.hpp` | Same migration approach. |
| `femx/system/SystemMatrix.hpp` | `femx/algebra/MatrixBuilder.hpp` plus matrix/operator adapter | Split builder and operator concepts. |
| `femx/system/SystemVector.hpp` | `femx/algebra/VectorBuilder.hpp` or backend vector | Consider whether this should stay backend-specific. |
| `femx/system/native/*` | `femx/algebra/backends/native/*` | Keep forwarding headers. |
| `femx/system/petsc/*` | `femx/algebra/backends/petsc/*` | Keep optional CMake target. |
| `femx/system/resolve/*` | `femx/algebra/backends/resolve/*` | Keep optional CMake target. |
| `femx/mesh/*` | `femx/fem/*` or `femx/mesh/*` as compatibility | The target tree folds mesh into fem-facing API. |
| `femx/fem/DofMap.hpp` | `femx/fem/DofMap.hpp` | Mostly stays. |
| `femx/assembly/DofLayout.hpp` | `femx/fem/DofLayout.hpp` | Layout is closer to FE space than generic assembly. |
| `femx/bc/*` | `femx/fem/BoundaryCondition.hpp` and related headers | Avoid making `problem` depend on `bc`. |
| `femx/assembly/ElementKernel.hpp` | `femx/assembly/Kernel.hpp` | Rename only after compatibility layer exists. |
| `femx/assembly/SystemAssembler.hpp` | `femx/assembly/Assembler.hpp` | Rename only after compatibility layer exists. |
| `femx/assembly/ElementResidualEquation.hpp` | `femx/assembly/FEMResidual.hpp` | This is a FEM implementation of the generic residual interface. |
| `femx/eq/ResidualEquation.hpp` | `femx/problem/Residual.hpp` | New interface should be centered on residual and linearization. |
| `femx/eq/TimeResidualEquation.hpp` | `femx/problem/TimeResidual.hpp` | Reduce method explosion later. |
| `femx/inverse/ObjectiveFunctional.hpp` | `femx/problem/Objective.hpp` | Objective is not necessarily inverse-specific. |
| `femx/inverse/ObservationOperator.hpp` | `femx/problem/Observation.hpp` | FEM-specific observation tools should not pollute generic objective code. |
| `femx/eq/*StateSolver*` | `femx/solve/Newton.hpp`, `femx/solve/TimeStepper.hpp` | Merge matrix/operator variants over time. |
| `femx/inverse/ReducedFunctional.hpp` | `femx/solve/ReducedFunctional.hpp` | Reduced objective is solve-level functionality. |
| `femx/inverse/DerivativeCheck.hpp` | `femx/solve/DerivativeCheck.hpp` | Generic utility. |
| `femx/inverse/petsc/TaoOptimizer.hpp` | `femx/optimize/TaoOptimizer.hpp` | PETSc adapter. Keep optional. |
| `femx/io/*` | `femx/io/*` | Mostly stays; trim to public target tree later. |

## 4. Migration principle: add new API first, then redirect old API

Do not start by deleting `system`, `eq`, or `inverse`. Instead:

1. Create the new directory and CMake target structure.
2. Move or copy minimal headers into the new structure.
3. Replace old headers with forwarding headers.
4. Update internal includes gradually.
5. Update examples and tests last.
6. Remove old headers only after all repository code uses the new paths and compatibility has been intentionally retired.

Forwarding header example:

```cpp
#pragma once

// Compatibility header. Prefer <femx/algebra/LinearOperator.hpp>.
#include <femx/algebra/LinearOperator.hpp>
```

If a namespace changes, provide aliases temporarily:

```cpp
namespace femx::system {
using LinearOperator = femx::algebra::LinearOperator;
using LinearSolver = femx::algebra::LinearSolver;
}
```

Only introduce namespace aliases when they do not create ODR or ambiguous type problems.

## 5. Phase 0: safety baseline

Before refactoring:

- [ ] Build the current project with the default configuration.
- [ ] Build with optional features disabled where possible:
  - [ ] `FEMX_ENABLE_RESOLVE=OFF`
  - [ ] `FEMX_ENABLE_HDF5=OFF`
  - [ ] `FEMX_ENABLE_PETSC=OFF`
- [ ] Run existing tests if available.
- [ ] Build at least one simple example that does not require PETSc.
- [ ] Build PETSc examples only if PETSc is available in the environment.
- [ ] Record the commands used in the PR or commit message.

Suggested commands:

```shell
cmake -S . -B build -DFEMX_ENABLE_RESOLVE=OFF -DFEMX_ENABLE_HDF5=OFF -DFEMX_ENABLE_PETSC=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

If optional dependencies are missing, do not block the refactor on them. State what could and could not be built.

## 6. Phase 1: create the new directory skeleton and CMake targets

Goal: add the target structure without changing behavior.

Tasks:

- [ ] Create directories:
  - [ ] `femx/core`
  - [ ] `femx/algebra`
  - [ ] `femx/algebra/backends/native`
  - [ ] `femx/algebra/backends/petsc`
  - [ ] `femx/algebra/backends/resolve`
  - [ ] `femx/problem`
  - [ ] `femx/solve`
  - [ ] `femx/optimize`
- [ ] Add `CMakeLists.txt` files for new targets.
- [ ] Keep old targets working:
  - [ ] `femx_common`
  - [ ] `femx_linalg`
  - [ ] `femx_system`
  - [ ] `femx_eq`
  - [ ] `femx_inverse`
- [ ] Add new aliases where safe:
  - [ ] `femx::core`
  - [ ] `femx::algebra`
  - [ ] `femx::problem`
  - [ ] `femx::solve`
  - [ ] `femx::optimize`
- [ ] Keep `femx::femx` linking all required components.

Acceptance criteria:

- [ ] The project configures.
- [ ] The project builds as before.
- [ ] No public include path is removed.
- [ ] The new directories exist even if they initially contain forwarding or wrapper headers only.

## 7. Phase 2: move core and algebra headers

Goal: clarify that vector, matrix, operator, and solver abstractions are algebra/backend concepts, not `system` concepts.

Tasks:

- [ ] Move `femx/common/Types.hpp` to `femx/core/Types.hpp`.
- [ ] Move `femx/common/Workspace.hpp` to `femx/core/Workspace.hpp`.
- [ ] Keep `femx/common/Types.hpp` and `femx/common/Workspace.hpp` as forwarding headers.
- [ ] Move algebraic linalg headers from `femx/linalg` to `femx/algebra`:
  - [ ] `Vector.hpp`
  - [ ] `VectorView.hpp` if still needed
  - [ ] `DenseMatrix.hpp`
  - [ ] `MatrixView.hpp` if still needed
  - [ ] `SparseMatrix.hpp`
  - [ ] `CsrPattern.hpp`
  - [ ] `IndexSetList.hpp`
- [ ] Keep `femx/linalg/*` forwarding headers.
- [ ] Move generic operator/solver headers:
  - [ ] `femx/system/LinearOperator.hpp` -> `femx/algebra/LinearOperator.hpp`
  - [ ] `femx/system/LinearSolver.hpp` -> `femx/algebra/LinearSolver.hpp`
- [ ] Add forwarding headers under `femx/system`.
- [ ] Update new headers to include from `femx/core` and `femx/algebra` rather than old paths.
- [ ] Do not change class behavior in this phase.

Acceptance criteria:

- [ ] Existing examples still compile with old include paths.
- [ ] New include paths compile in a small smoke test.
- [ ] No solver logic is changed.

Smoke-test snippet:

```cpp
#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/LinearOperator.hpp>
#include <femx/algebra/LinearSolver.hpp>

int main() {
  femx::Vector<femx::Real> x(3);
  return x.size() == 3 ? 0 : 1;
}
```

## 8. Phase 3: split `SystemMatrix` into operator and builder concepts

Goal: stop making every assembled matrix conceptually equal to a linear operator plus mutable assembly target.

Current issue:

- `SystemMatrix` currently combines:
  - matrix dimensions
  - operator application
  - mutable assembly methods
  - finalization
- This causes solver, assembler, and backend code to know too much about each other.

New target concepts:

```cpp
namespace femx::algebra {

class LinearOperator {
public:
  virtual ~LinearOperator() = default;
  virtual Index numRows() const = 0;
  virtual Index numCols() const = 0;
  virtual void apply(const Vector<Real>& x, Vector<Real>& y) const = 0;
  virtual void applyT(const Vector<Real>& x, Vector<Real>& y) const = 0;
};

class MatrixBuilder {
public:
  virtual ~MatrixBuilder() = default;
  virtual void resize(Index rows, Index cols) = 0;
  virtual void setZero() = 0;
  virtual void set(Index row, Index col, Real value) = 0;
  virtual void add(Index row, Index col, Real value) = 0;
  virtual void finalize() = 0;
};

class MatrixOperator : public LinearOperator, public MatrixBuilder {
public:
  ~MatrixOperator() override = default;
};

} // namespace femx::algebra
```

Tasks:

- [ ] Add `femx/algebra/MatrixBuilder.hpp`.
- [ ] Add `femx/algebra/MatrixOperator.hpp` if useful.
- [ ] Change backend matrix implementations to inherit `MatrixOperator` or both `LinearOperator` and `MatrixBuilder`.
- [ ] Keep `system::SystemMatrix` as a compatibility alias or thin derived class temporarily.
- [ ] Keep `SystemVector` compatibility for now. Do not solve vector-builder design in this phase unless trivial.
- [ ] Update assemblers to depend on `MatrixBuilder` where they only assemble.
- [ ] Update solvers to depend on `LinearOperator` where they only solve.

Acceptance criteria:

- [ ] Assembly code does not require full `SystemMatrix` when only builder methods are needed.
- [ ] Linear solvers do not require mutable matrix methods.
- [ ] Old code using `system::SystemMatrix` still compiles.

## 9. Phase 4: introduce `problem::Residual` and `problem::Linearization`

Goal: make equations expose residuals and linearizations without multiplying `Matrix...` and `Operator...` classes.

Add these new concepts first, without deleting existing `eq` classes:

```cpp
namespace femx::problem {

struct Dimensions {
  Index num_states = 0;
  Index num_params = 0;
  Index num_residuals = 0;
};

class Linearization {
public:
  virtual ~Linearization() = default;

  virtual const algebra::LinearOperator& stateJacobian() const = 0;
  virtual const algebra::LinearOperator& paramJacobian() const = 0;
};

class Residual {
public:
  virtual ~Residual() = default;

  virtual Dimensions dimensions() const = 0;

  virtual void residual(const Vector<Real>& state,
                        const Vector<Real>& prm,
                        Vector<Real>& out) const = 0;

  virtual void linearize(const Vector<Real>& state,
                         const Vector<Real>& prm,
                         Linearization& out) const = 0;
};

} // namespace femx::problem
```

Implementation guidance:

- `Linearization` may own or reference assembled matrices.
- Avoid returning raw pointers to temporary matrices.
- If ownership is hard, start with a concrete `MatrixLinearization` that stores references to preallocated backend matrices.
- Keep the sign convention identical to current residual and adjoint code.
- Do not remove current `eq::ResidualEquation` yet.

Tasks:

- [ ] Add `femx/problem/Residual.hpp`.
- [ ] Add `femx/problem/Linearization.hpp`.
- [ ] Add adapter from existing `eq::MatrixResidualEquation` to new `problem::Residual` if useful.
- [ ] Add adapter from existing `eq::ResidualEquation` to new `problem::Residual` if possible.
- [ ] Add tests or smoke examples for a small 2x2 residual problem.

Acceptance criteria:

- [ ] A small residual problem can be solved through the new interface.
- [ ] Existing `eq` code still builds.
- [ ] No old example behavior changes.

## 10. Phase 5: unify `MatrixNewtonStateSolver` and `OperatorNewtonStateSolver`

Goal: replace matrix/operator solver duplication with a single Newton state solver using `problem::Residual` and `problem::Linearization`.

New target:

```cpp
namespace femx::solve {

struct NewtonOptions {
  Index max_its = 20;
  Real residual_tolerance = 1.0e-10;
  Real step_tolerance = 0.0;
};

class Newton {
public:
  Newton(const problem::Residual& problem,
         algebra::LinearSolver& linear_solver);

  NewtonOptions& options();
  const NewtonOptions& options() const;

  void setInitialState(const Vector<Real>& state);
  void clearInitialState();

  void solve(const Vector<Real>& prm, Vector<Real>& state);
};

} // namespace femx::solve
```

Tasks:

- [ ] Add `femx/solve/Newton.hpp` and implementation.
- [ ] Implement using `problem::Residual::linearize`.
- [ ] Use `linear_solver.solve(linearization.stateJacobian(), rhs, step)`.
- [ ] Keep old classes as wrappers:
  - [ ] `eq::MatrixNewtonStateSolver`
  - [ ] `eq::OperatorNewtonStateSolver`
  - [ ] `eq::MatrixLinearStateSolver`, if possible
- [ ] Avoid changing public behavior of old classes.
- [ ] Add deprecation comments, but do not use compiler deprecation attributes until examples are migrated.

Acceptance criteria:

- [ ] New `solve::Newton` solves the existing small linear-control example problem or an equivalent test problem.
- [ ] Old Newton solver APIs still compile.
- [ ] No duplicate Newton implementation remains if wrappers are feasible.

## 11. Phase 6: simplify adjoint/reduced functional design

Goal: make adjoint solve an operation over the linearized state Jacobian, not a separate class hierarchy.

Current issue:

- `inverse::MatrixAdjointSolver` and `inverse::OperatorAdjointSolver` differ mainly in how they access `R_u`.
- Once `problem::Linearization` exposes `stateJacobian()`, both can be replaced by `linear_solver.solveT(...)`.

New target:

```cpp
namespace femx::solve {

class ReducedFunctional {
public:
  ReducedFunctional(problem::Residual& problem,
                    problem::Objective& objective,
                    Newton& state_solver,
                    algebra::LinearSolver& adjoint_linear_solver);

  Index numParams() const;
  Real value(const Vector<Real>& prm);
  void grad(const Vector<Real>& prm, Vector<Real>& out);
  Real valueGrad(const Vector<Real>& prm, Vector<Real>& grad_out);
};

} // namespace femx::solve
```

Tasks:

- [ ] Move or wrap `inverse::ObjectiveFunctional` as `problem::Objective`.
- [ ] Move or wrap `inverse::ReducedFunctional` as `solve::ReducedFunctional`.
- [ ] Implement `solve::ReducedFunctional` using:
  - [ ] state solve
  - [ ] objective state gradient
  - [ ] residual linearization
  - [ ] transpose solve against `stateJacobian()`
  - [ ] objective parameter gradient
  - [ ] transpose apply of parameter Jacobian
- [ ] Preserve current gradient formula exactly.
- [ ] Keep existing `inverse::AdjointReducedFunctional` as a wrapper if possible.
- [ ] Keep `inverse::MatrixAdjointSolver` and `inverse::OperatorAdjointSolver` during transition, but mark them as compatibility classes in comments.

Acceptance criteria:

- [ ] Existing inverse linear-control example can be migrated to the new API in a separate step.
- [ ] Old inverse example still builds before migration.
- [ ] Gradient check passes if available.

## 12. Phase 7: move objective and observation out of `inverse`

Goal: make objective and observation generic problem concepts.

Tasks:

- [ ] Add `femx/problem/Objective.hpp` with the generic scalar objective interface.
- [ ] Add `femx/problem/Observation.hpp` with the generic observation interface.
- [ ] Move or forward:
  - [ ] `inverse/ObjectiveFunctional.hpp` -> `problem/Objective.hpp`
  - [ ] `inverse/ObservationOperator.hpp` -> `problem/Observation.hpp`
- [ ] Keep least-squares and regularization classes either in `solve` or `problem/objectives` depending on how generic they are.
- [ ] Move FEM-specific observation helpers out of the generic objective core.

Suggested split:

```text
femx/problem/Objective.hpp
femx/problem/Observation.hpp
femx/solve/ReducedFunctional.hpp
femx/solve/DerivativeCheck.hpp
femx/fem/ObservationGrid.hpp          # only if FEM-specific
femx/fem/TimePointSampler.hpp         # only if FEM-specific
```

Acceptance criteria:

- [ ] Generic objective code does not include FEM headers.
- [ ] FEM observation helpers may include FEM headers.
- [ ] Old `inverse/*` include paths still work.

## 13. Phase 8: reorganize FEM, BC, and assembly

Goal: make FEM-facing data structures and assembly utilities easier to find.

Tasks:

- [ ] Decide whether `mesh` remains as a compatibility folder or is folded into `fem` publicly.
- [ ] Move or forward these into `fem`:
  - [ ] `Mesh.hpp`
  - [ ] `Cell.hpp`
  - [ ] `FESpace.hpp`
  - [ ] `MixedFESpace.hpp`
  - [ ] `FiniteElement.hpp`
  - [ ] `GaussQuadrature.hpp` -> public alias `Quadrature.hpp`
  - [ ] `DofMap.hpp`
  - [ ] `DofLayout.hpp`
  - [ ] `BoundaryCondition.hpp` or equivalent BC wrapper
- [ ] Rename assembly concepts only after forwarding headers exist:
  - [ ] `ElementKernel.hpp` -> `Kernel.hpp`
  - [ ] `SystemAssembler.hpp` -> `Assembler.hpp`
  - [ ] `ElementResidualEquation.hpp` -> `FEMResidual.hpp`
- [ ] Ensure assembly depends on `algebra::MatrixBuilder`, not backend-specific matrices.
- [ ] Ensure generic `problem` code does not depend on `fem` or `bc`.

Acceptance criteria:

- [ ] A FEM residual can be assembled through the new `assembly::FEMResidual` or equivalent.
- [ ] Old assembly include paths still compile.
- [ ] Dependency from generic `problem` to `bc` is removed.

## 14. Phase 9: simplify time-dependent interfaces

Goal: stop creating separate methods/classes for every combination of time, matrix/operator, next-state, previous-state, and parameter derivative.

Preferred direction:

```cpp
namespace femx::problem {

struct TimeContext {
  Index step = 0;
  const Vector<Real>* previous_state = nullptr;
  const Vector<Real>* next_state = nullptr;
  const Vector<Real>* prm = nullptr;
};

enum class VariableBlock {
  PreviousState,
  NextState,
  Parameter
};

class TimeResidual {
public:
  virtual ~TimeResidual() = default;

  virtual TimeDimensions dimensions() const = 0;

  virtual void residual(const TimeContext& ctx,
                        Vector<Real>& out) const = 0;

  virtual void applyJacobian(const TimeContext& ctx,
                             VariableBlock wrt,
                             const Vector<Real>& dir,
                             Vector<Real>& out) const = 0;

  virtual void applyJacobianT(const TimeContext& ctx,
                              VariableBlock wrt,
                              const Vector<Real>& adjoint,
                              Vector<Real>& out) const = 0;
};

} // namespace femx::problem
```

Optional matrix path:

```cpp
virtual bool assembleJacobian(const TimeContext& ctx,
                              VariableBlock wrt,
                              algebra::MatrixBuilder& out) const;
```

Tasks:

- [ ] Add the new `problem::TimeResidual` interface.
- [ ] Add adapters from old `eq::TimeResidualEquation` and `eq::TimeMatrixResidualEquation`.
- [ ] Implement `solve::TimeStepper` against the new interface.
- [ ] Keep old time solver classes temporarily as wrappers.
- [ ] Avoid migrating all applications in this phase unless required.

Acceptance criteria:

- [ ] New time interface can represent existing `next`, `previous`, and `parameter` Jacobian operations.
- [ ] Old time-dependent code still compiles.
- [ ] No sign or transpose convention changes.

## 15. Phase 10: move PETSc TAO adapter to `optimize`

Goal: keep optimization package adapters separate from reduced-functional theory.

Tasks:

- [ ] Move or forward `inverse/petsc/TaoOptimizer.hpp` to `optimize/TaoOptimizer.hpp`.
- [ ] Move or forward `TaoReducedFunctionalAdapter.hpp` to `optimize`.
- [ ] Ensure `femx::optimize` is only built when PETSc support is enabled.
- [ ] Keep `femx::inverse_petsc` as compatibility target or alias if feasible.
- [ ] Update examples only after compatibility is confirmed.

Acceptance criteria:

- [ ] PETSc-free builds do not require TAO headers.
- [ ] PETSc builds still compile the TAO optimizer adapter.
- [ ] Existing inverse PETSc example can be migrated to the new include path in a separate commit.

## 16. Phase 11: update examples after compatibility exists

Goal: demonstrate the new API without breaking old paths too early.

Start with the smallest inverse example.

Old style:

```cpp
LinearResidualEquation    res_eq;
system::PETScSystemMatrix state_jac;
system::PETScSystemMatrix adj_state_jac;
system::KspLinearSolver   lin_solver;

// MatrixNewtonStateSolver + MatrixAdjointSolver + AdjointReducedFunctional
```

Target style:

```cpp
LinearControlProblem problem;
system_or_algebra_backend backend;
solve::Newton state_solver(problem, lin_solver);
solve::ReducedFunctional functional(problem, objective, state_solver, lin_solver);
optimize::TaoOptimizer optimizer(functional);
```

Tasks:

- [ ] Add one new example using the new API.
- [ ] Keep the old example temporarily or update it after the new one passes.
- [ ] Prefer a minimal 2x2 problem before migrating Navier-Stokes applications.
- [ ] Once the new example is stable, migrate larger examples.

Acceptance criteria:

- [ ] New example has fewer user-visible objects than the old one.
- [ ] New example obtains the same objective and gradient behavior as the old one.
- [ ] README documents the new preferred include paths.

## 17. Phase 12: cleanup and deprecation

Only do this after all internal code and examples use the new paths.

Tasks:

- [ ] Add comments to old forwarding headers saying they are compatibility headers.
- [ ] Optionally add `[[deprecated]]` aliases only after examples no longer use old paths.
- [ ] Remove obsolete duplicate implementations.
- [ ] Keep compatibility headers for at least one transition period.
- [ ] Update README component target list to show the simplified public targets.

Suggested public target list:

```text
femx::femx
femx::core
femx::algebra
femx::fem
femx::assembly
femx::problem
femx::solve
femx::optimize       # only when PETSc/TAO is enabled
femx::io
```

If that is still too many, expose only:

```text
femx::femx
femx::petsc          # optional backend/optimizer bundle
femx::resolve        # optional backend bundle
```

## 18. Include and namespace policy

Preferred new include style:

```cpp
#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/solve/Newton.hpp>
```

Namespace policy:

- Keep class names stable before changing namespaces.
- Avoid doing file moves and namespace moves in the same phase.
- If namespaces are changed, provide aliases for old namespaces.
- Avoid deep namespaces like `femx::algebra::backends::petsc` in user-facing examples unless needed. A namespace such as `femx::petsc` may be more readable.

Suggested eventual namespaces:

```cpp
namespace femx;
namespace femx::algebra;
namespace femx::problem;
namespace femx::solve;
namespace femx::petsc;
namespace femx::resolve;
```

## 19. CMake policy

- Keep `femx_config` or equivalent central config target.
- Keep optional dependency flags centralized at the root.
- New targets should link only what they need.
- Avoid circular links.
- Compatibility targets may link to new targets.
- Do not make `problem` link against `fem`, `assembly`, `bc`, PETSc, ReSolve, or HDF5.
- Do not make `algebra` link against PETSc or ReSolve directly; backend subtargets may do so.

Suggested target dependencies:

```text
femx_core -> femx_config
femx_algebra -> femx_core
femx_algebra_native -> femx_algebra
femx_algebra_petsc -> femx_algebra + PETSc + MPI
femx_algebra_resolve -> femx_algebra + ReSolve
femx_fem -> femx_core + femx_algebra
femx_assembly -> femx_fem + femx_problem + femx_algebra
femx_problem -> femx_core + femx_algebra
femx_solve -> femx_problem + femx_algebra
femx_optimize -> femx_solve + PETSc + MPI
femx_io -> femx_core + femx_algebra + femx_fem + optional HDF5
```

## 20. Testing checklist for every phase

Run as many of these as the environment supports:

```shell
cmake -S . -B build-basic \
  -DFEMX_ENABLE_RESOLVE=OFF \
  -DFEMX_ENABLE_HDF5=OFF \
  -DFEMX_ENABLE_PETSC=OFF
cmake --build build-basic
ctest --test-dir build-basic --output-on-failure
```

If PETSc is available:

```shell
cmake -S . -B build-petsc \
  -DFEMX_ENABLE_PETSC=ON \
  -DFEMX_ENABLE_HDF5=OFF \
  -DFEMX_ENABLE_RESOLVE=OFF
cmake --build build-petsc
ctest --test-dir build-petsc --output-on-failure
```

If ReSolve is available:

```shell
cmake -S . -B build-resolve \
  -DFEMX_ENABLE_RESOLVE=ON \
  -DFEMX_ENABLE_PETSC=OFF
cmake --build build-resolve
```

## 21. Things Codex should avoid

- Do not delete old directories before all code is migrated.
- Do not rename everything in one huge patch.
- Do not mix numerical algorithm changes with file layout changes.
- Do not change residual or adjoint sign conventions.
- Do not make `problem` depend on `assembly` or `fem`.
- Do not make `algebra` depend on PETSc or ReSolve directly.
- Do not expose PETSc types in generic `solve` APIs.
- Do not silently skip broken examples; either fix them or document why they were not built.
- Do not replace virtual interfaces with templates everywhere in this refactor. Template-based optimization can be a later step.
- Do not introduce a new package manager or dependency discovery mechanism.

## 22. Suggested first Codex task

Use this as the first concrete prompt to Codex:

```text
Refactor femx incrementally toward the architecture described in CODEX_REFACTOR_PLAN.md.

Start with Phase 1 and Phase 2 only:
1. Create the new directory skeleton for core, algebra, algebra/backends, problem, solve, and optimize.
2. Move or copy core and algebra headers into the new locations.
3. Add forwarding headers in the old locations so existing includes keep working.
4. Add or update CMake targets so femx::femx still works.
5. Do not change solver behavior or numerical algorithms.
6. Do not migrate examples yet unless required for compilation.
7. Build with optional dependencies disabled if the local environment lacks PETSc, HDF5, or ReSolve.
8. Summarize exactly which headers were moved, which forwarding headers were added, and which build commands were run.
```

Expected result of the first task:

- New include paths exist for core and algebra.
- Old include paths still compile.
- CMake still configures and builds.
- No `MatrixNewtonStateSolver`, `OperatorNewtonStateSolver`, `AdjointReducedFunctional`, or time-solver logic is changed yet.

## 23. Suggested second Codex task

Use this only after the first task is merged or stable:

```text
Continue the refactor using CODEX_REFACTOR_PLAN.md.

Implement Phase 3:
1. Add algebra::MatrixBuilder and, if useful, algebra::MatrixOperator.
2. Make backend matrix classes implement the new builder/operator interfaces.
3. Keep system::SystemMatrix as a compatibility layer.
4. Update assemblers to depend on MatrixBuilder where they only assemble.
5. Update solvers to depend on LinearOperator where they only solve.
6. Do not change numerical behavior.
7. Keep old include paths and old class names compiling.
8. Build and report the exact commands and results.
```

## 24. Suggested third Codex task

Use this after the builder/operator split is stable:

```text
Continue the refactor using CODEX_REFACTOR_PLAN.md.

Implement Phase 4 and prepare Phase 5:
1. Add problem::Residual, problem::Dimensions, and problem::Linearization.
2. Add adapters from the existing eq::ResidualEquation and eq::MatrixResidualEquation where practical.
3. Add a small smoke test or example with a 2x2 residual problem.
4. Do not remove old eq classes.
5. Do not rewrite inverse or time-dependent code yet.
6. Build and report the exact commands and results.
```

## 25. Done definition for the full migration

The full migration is done when:

- [ ] The preferred public headers match the target tree.
- [ ] `system`, `eq`, and `inverse` are either removed or only compatibility layers.
- [ ] Matrix/operator Newton duplication is removed or fully hidden behind wrappers.
- [ ] Matrix/operator adjoint duplication is removed or fully hidden behind wrappers.
- [ ] Time residual method explosion is replaced by a block-based Jacobian interface or equivalent.
- [ ] Generic `problem` and `solve` code have no FEM, BC, PETSc, ReSolve, or HDF5 dependencies.
- [ ] Backends are isolated under `algebra/backends` or equivalent.
- [ ] Examples use the simplified API.
- [ ] README shows the new structure and preferred include paths.
- [ ] All available tests pass.
```
