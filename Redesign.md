# Backend / MemorySpace redesign

## 目的

Host と Device で同じ数値アルゴリズムを二重実装せず、違いを小さな
backend 境界へ閉じ込める。既存 API との互換性より、最終的な構造の
単純さを優先する。

主な目標は次のとおり。

- `Space` はデータの配置だけを表す。
- `Backend` は行列・ベクトル・context と基本演算の組を表す。
- 時間積分、adjoint、制約処理などのアルゴリズムは `Backend` に対して
  一度だけ実装する。
- 行列・ベクトルは小さなデータ型とし、solver や CUDA descriptor を
  所有させない。
- Host/Device 間の copy と CUDA 同期は明示的なままにする。
- PETSc を単なる `Host` の特殊例にせず、独立した backend として扱う。

## 用語と責務

### MemorySpace

`MemorySpace` はポインタがどこを指すか、どこで所有されるかだけを表す。

```cpp
Vector<MemorySpace::Host>
Vector<MemorySpace::Device>
CsrGraph<MemorySpace::Host>
CsrGraph<MemorySpace::Device>
CsrMatrix<MemorySpace::Host>
CsrMatrix<MemorySpace::Device>
```

`Vector`、`CsrGraph`、`CsrMatrix`、`AssemblyMap`、`Trajectory` などの
データ型に使用する。

### Backend

`Backend` は一つの計算経路で使用する型と基本操作をまとめる。
C++17 なので concept は使わず、小さな traits/policy と `static_assert`
で必要な interface を検査する。

概念的には次の型を提供する。

```cpp
struct HostCsrBackend
{
  static constexpr MemorySpace space = MemorySpace::Host;
  using Vec = HostVector;
  using Mat = HostCsrMatrix;
  using Ctx = CpuContext;
};

struct CudaCsrBackend
{
  static constexpr MemorySpace space = MemorySpace::Device;
  using Vec = DeviceVector;
  using Mat = DeviceCsrMatrix;
  using Ctx = CudaContext;
};

struct PetscBackend
{
  // 時間積分の公開stateはHostVector。PETSc Vecへの変換はKSP境界だけ。
  using Vec = HostVector;
  using Mat = PETScOperator;
  using Ctx = PetscContext;
};
```

実際に必要な view や trajectory 型も、必要になった段階で backend に
追加する。backend に無関係な domain 型は追加しない。

### Solver

ReSolve、KSP、dense solve は backend そのものではなく、対応する backend
上で動く solver 実装である。solver は抽象的な `LinearOperator` ではなく、
backend の具体的な行列型を受け取る。

```cpp
template <class Backend>
class LinearSolver
{
public:
  using Matrix = typename Backend::Mat;
  using Vector = typename Backend::Vec;
  using Context = typename Backend::Ctx;

  virtual ~LinearSolver() = default;
  virtual void solve(const Matrix& mat,
                     const Vector& rhs,
                     Vector&       sol,
                     Context&      ctx) = 0;
  virtual void solveT(const Matrix& mat,
                      const Vector& rhs,
                      Vector&       sol,
                      Context&      ctx) = 0;
};
```

Host の空の `CpuContext` も引数に含め、共通アルゴリズムの呼び出し形を
揃える。必要なら便利な Host overload を追加するが、共通実装は同じ形を
使用する。

## ReSolve から採用する考え方

ReSolve は行列・ベクトルをデータ管理に限定し、SpMV、BLAS、変換などを
`MatrixHandler` と `VectorHandler` に分離している。CUDA/HIP の handle、
descriptor、buffer は workspace に置かれ、solver は同じ `setMatrix()` と
`solve()` API を使う。

femx では次を採用する。

- データと演算を分離する。
- backend 固有の handle、descriptor、temporary buffer を再利用する。
- solver は具体的な行列データを直接受け取る。
- 高水準のアルゴリズムから CPU/CUDA 分岐を除く。

一方、次は採用しない。

- 一つの行列・ベクトルに Host/Device 両方のポインタを持たせない。
- dirty flag による暗黙の同期を導入しない。
- 各演算で runtime の `MemorySpace` switch を行わない。
- context を隠して default stream に固定しない。

femx では現在の `Type<Space>` と明示的な `copy(..., ctx)` を維持し、
backend は compile-time に選択する。

## linalg の目標

### CsrMatrix

`CsrMatrix` は最終的に graph と数値だけを持つ。

```cpp
template <MemorySpace Space>
class CsrMatrix
{
public:
  using Graph = CsrGraph<Space>;
  using Vals  = Vector<Space>;

  explicit CsrMatrix(Graph graph);

  Index rows() const;
  Index cols() const;
  Index nnz() const;

  const Graph& graph() const;
  Vals&        vals();
  const Vals&  vals() const;

private:
  Graph graph_;
  Vals  vals_;
};
```

次を `CsrMatrix` から除く。

- solver/backend state
- `detail::CsrBackendAccess`
- cuSPARSE descriptor と workspace
- solver/factorization の状態
- Host/Device 固有の処理分岐

`apply`、`applyT`、`copy` などの低レベル API は自由関数のままでよい。

```cpp
apply(host_mat, x, y);
apply(dev_mat, x, y, ctx);
copy(host_mat, dev_mat, ctx);
```

共通アルゴリズムからは backend policy を介して同じ形で呼び出す。
CUDA descriptor と buffer は `CudaContext`、CUDA workspace、または
solver-owned workspace に移す。matrix identity、`layoutId`、値ポインタを
記録し、構造が変わらない限り再利用する。行列自身は mutable な backend
cache を持たない。

### DenseMatrix

`DenseMatrix` も Host 用のデータ型に戻す。`MatrixOperator` を継承せず、
`apply` と `applyT` は既存の自由関数を使用する。小規模 dense solve は
`DenseBackend` または独立したテスト用 solver として扱う。

### 削除する operator 階層

移行完了後に次を削除する。

- `LinearOperator`
- `MatrixOperator`
- `CsrOperator`
- `MatrixLinearization`
- `AssemblyMatrix`

ReSolve adapter の `dynamic_cast<CsrOperator>` も削除し、
`HostCsrMatrix` または `DeviceCsrMatrix` を直接 bind する。

PETSc 移行前に operator 階層を削除すると PETSc 経路が壊れるため、削除は
最後に行う。移行中だけ旧 interface と新 interface の adapter を許可する。

## assembly の目標

`AssemblyMap<Space>` と `BoundaryMap<Space>` は純粋なデータとして維持する。
有限要素演算と制約操作は backend operation とする。

```cpp
assemble(map, elem_op, ctx, res, jac);
applyConstraints(boundary, ctx, jac, rhs);
```

CPU要素ループ、CUDA kernel、PETScのentry insertionは個別実装として残す。
その上の検査、zero、時間contextの構築、制約処理の順序は共通化する。

`AssemblyMatrix` が持っている高速な `jac_map` 経路は失わない。
`AssemblyMap` がすでにlocal JacobianからCSR位置へのmapを持つため、
Host/CUDA CSR backendでは`CsrMatrix<Space>`へ直接assembleする。

残す低レベル実装は次のとおり。

- CPU element loop
- CUDA assembly kernel
- CUDA atomic accumulation
- PETSc insertion/finalize
- Host/Device boundary kernels

## state の目標

### TimeResidual

Host/Device の完全特殊化を一つの backend template に置き換える。

```cpp
template <class Backend>
class TimeResidual;

using HostTimeResidual = TimeResidual<HostCsrBackend>;
using DeviceTimeResidual = TimeResidual<CudaCsrBackend>;
```

`Backend::Mat`、`Backend::Vec`、`Backend::Ctx`を使って次のinterfaceを
共通化する。

- `initialState`
- `addInitialJacT`
- `res`
- `assemble`
- `applyJac`
- `applyJacT`
- `assembleJac`
- `prepareLinearSolve`

### TimeIntegrator

時間ループは一つだけ実装する。

```cpp
template <class Backend>
class TimeIntegrator;
```

共通化する処理は次のとおり。

1. 初期状態を評価する。
2. history を初期化する。
3. residual と next-state Jacobian をassembleする。
4. Dirichlet処理などの`prepareLinearSolve`を呼ぶ。
5. backend solverで解く。
6. historyとtrajectoryを更新する。
7. observerと統計を更新する。

backendへ残す処理はzero/copy、assembly、solve、同期、必要なevent計測だけと
する。旧`TimeIntegrator.cpp`と`TimeIntegrator.cu`の制御コードを一つにする。

## model と制約の目標

### ConstrainedTimeResidual

Host/Device特殊化を`ConstrainedTimeResidual<Backend>`へ統合する。
境界値の計算、初期状態、Jacobian転置への寄与、処理順序は共通化し、
scatter、row replacement、column eliminationだけをbackendへ渡す。

### Navier-Stokes

`NavierOperator<Space>`と共有element dataは維持する。現在のHost residualと
`DeviceNavierResidual`を`NavierResidual<Backend>`へ統合する。

共通部分は次のとおり。

- dimensionとcontextの検査
- `VariableBlock`の処理
- history/next stateの意味
- residual/Jacobian assemblyの呼び出し
- 制約decoratorとの接続

CPU element loop、CUDA kernel、MPI reductionはbackend固有処理として残す。

## inverse の目標

`TimeReducedFunctional<MemorySpace::Host>`とDevice特殊化を、
`TimeReducedFunctional<Backend>`の一つのadjointアルゴリズムへ統合する。

共通化する処理は次のとおり。

1. forward trajectoryを計算する。
2. objective valueと各時間levelのgradientを計算する。
3. 終端adjointを作る。
4. 時間を逆向きに走査する。
5. history/next-state Jacobian転置を適用する。
6. adjoint systemを解く。
7. parameterおよびinitial-state gradientを加算する。

backend固有なのはvector operation、Jacobian assembly、transpose solve、copy、
同期だけとする。Objectiveの数式やadjointの順序はbackendへ入れない。

## PETSc の扱い

`MemorySpace::Host`とPETSc backendを同一視しない。PETScはmatrix/vectorの
所有、MPI communicator、assembly、solveがCSR Host backendと異なる。

`PetscBackend`の時間積分APIは次を使用する。

- replicated state用の`HostVector`
- `PETScOperator`
- communicator/context
- PETSc assembly primitives
- `KspLinearSolver<PetscBackend>`

分散layoutは`PETScOperator`がcommunicatorから直接決定する。layoutを得る
ためだけの公開`PETScVector`は置かず、native `Vec`はKSP/TAO境界の短命な
workspaceに限定する。時間trajectoryを分散化するまでは、KSP境界でだけ
Host stateとPETSc Vecを明示変換する。PETSc行列をHost CSRへ変換する経路は
作らない。

## 目標ファイル構成

概略として次を目指す。

```text
femx/
├── common/
│   ├── Context.hpp
│   ├── Checks.hpp
│   └── Types.hpp
├── linalg/
│   ├── Backend.hpp
│   ├── View.hpp
│   ├── Vector.hpp
│   ├── Vector.cu
│   ├── CsrGraph.hpp
│   ├── CsrMatrix.hpp
│   ├── CsrMatrix.cpp
│   ├── CsrMatrix.cu
│   ├── Dense.hpp
│   ├── Dense.cpp
│   ├── LinearSolver.hpp
│   ├── petsc/
│   └── resolve/
├── assembly/
│   ├── AssemblyMap.hpp
│   ├── AssemblyMap.cpp
│   ├── Assembly.hpp
│   ├── Assembly.cpp
│   ├── CudaAssembly.hpp
│   ├── BoundaryMap.hpp
│   ├── BoundaryMap.cpp
│   ├── BoundaryMap.cu
│   └── ConstrainedTimeResidual.hpp
├── state/
│   ├── Residual.hpp
│   ├── StateSolver.hpp
│   ├── TimeResidual.hpp
│   ├── TimeTrajectory.hpp
│   └── TimeIntegrator.hpp
├── inverse/
│   ├── ReducedFunctional.hpp
│   ├── TimeObjective.hpp
│   └── TimeReducedFunctional.hpp
└── model/ns/
    ├── Components.hpp
    ├── NavierStokesModel.hpp
    └── backend固有のkernel実装
```

テンプレート定義を公開headerへ大量に置く必要がある場合は、対応する
`Detail.hpp`と明示的instantiationを使用する。Host/Deviceの制御コードを
再び別実装にしない。

## 実装手順

### Phase 1: backendの最小定義

1. `HostCsrBackend`と`CudaCsrBackend`を追加する。
2. `Vec`、`Mat`、`Ctx`、`space`だけを最初に定義する。
3. 既存のfree functionを呼ぶ小さなbackend primitiveを用意する。
4. runtime `MemorySpace` dispatchや巨大な`DeviceOps`は作らない。
5. backend単体のcompile-time testを追加する。

### Phase 2: linalgとsolver

1. `LinearSolver<Backend>`を具体的な`Backend::Mat`/`Vec`で定義する。
2. ReSolve Host/Device実装を新interfaceへ移す。
3. ReSolve Host pathの`CsrOperator` dynamic castを削除する。
4. cuSPARSE descriptor/workspaceを`CsrMatrix`外へ移す。
5. `DenseMatrix`をstorageとfree operationへ整理する。
6. この段階では旧`LinearOperator` adapterを一時的に残してよい。

### Phase 3: assemblyとstate

1. CSR assemblyの出力を`CsrMatrix<Space>`へ直接統一する。
2. `TimeResidual<Backend>`を実装する。
3. `TimeIntegrator<Backend>`へHost/Device時間ループを統合する。
4. `ConstrainedTimeResidual<Backend>`を統合する。
5. Host/CUDAの既存テストを新APIへ書き換える。

### Phase 4: model

1. Navier residualを`NavierResidual<Backend>`へ統合する。
2. Host/CUDAのelement assembly呼び出しだけをbackendに残す。
3. `makeDeviceTimeResidual`のようなbackend固有factoryを共通factoryへ置き換える。
4. model-owned contextとsolver-owned contextの所有関係を一つに決める。

### Phase 5: inverse

1. `TimeReducedFunctional<Backend>`へforward/adjoint制御を統合する。
2. objective planのHost/Device差をbackend primitiveへ移す。
3. transpose matrix cacheはsolverまたはbackend workspaceへ移す。
4. gradientとfinite-difference testをHost/CUDA双方で実行する。

### Phase 6: PETScと旧階層の削除

1. `PetscBackend`を実装する。
2. KSP solverとPETSc assemblyを新interfaceへ移す。
3. `LinearOperator`、`MatrixOperator`、`CsrOperator`を削除する。
4. `AssemblyMatrix`と`MatrixLinearization`を削除する。
5. 一時adapterと旧API用overloadを削除する。
6. 不要になった`.cpp`/`.cu`とCMake source entryを削除する。

各phaseはbuild可能な状態で完了させる。一時adapterは用途と削除phaseを明記し、
新旧二系統の実装を増やさない。

## 実装状況（2026-07-19）

完了した項目:

- `HostCsrBackend`、`CudaCsrBackend`、`PetscBackend`とcompile-time検査を追加。
- `LinearSolver<Backend>`を追加し、Host CSR、CUDA CSR、PETScの具体型を直接
  受け取る`solve` / `solveT`へ統一。
- ReSolveの公開`setOperator`とsolver-bound overloadを削除。CUDA転置solveは
  cuSPARSEのCSR-to-CSC変換を使用し、Host graphへの依存を削除。
- `CsrMatrix`をgraphとvaluesだけのデータ型に変更。cuSPARSE descriptorと
  SpMV bufferは`CudaContext`側、転置workspaceはsolver側へ移動。
- Host/CUDA/PETScの`TimeIntegrator`制御ループを一つのheader実装へ統合し、
  backend別のintegrator sourceを削除。
- Host/CUDAのforward/adjoint制御を一つの`AdjointCore`へ統合し、
  `TimeReducedFunctional.cu`を削除。
- CUDA kernelを持たないDevice制約処理をHost側実装と同じ
  `ConstrainedTimeResidual.cpp`へ統合し、専用`.cu`を削除。
- PETSc KSPに`PetscContext`付きの具体backend APIを追加し、contextなしの
  PETSc専用overloadを削除。
- `AssemblyMap`から`HostCsrMatrix`へ直接加算・行置換・列消去するassembly
  primitiveを`Assembly.cpp`へ集約し、`AssemblyMatrix`を削除。
- Navier Host residualの`assemble` / `assembleJac`要素ループを一つの
  template実装へ統合。Host CSRとPETScの具体型dispatchはループ外で
  一度だけ行い、PETScはentry単位ではなくelement block単位で挿入する。
- `TimeResidual<Backend>`と`TimeLinearization<Backend>`へ公開APIを移し、
  matrix、vector、context、graphをbackendの具体型へ統一。
- `ConstrainedTimeResidual<Backend>`を追加し、Host CSRとPETScは同じtemplate
  実装を使用。CUDAも同じ公開contractを使い、residual内にcontextを所有しない。
- 初期状態の評価と転置微分を`TimeResidual<Backend>`へ移し、Host integratorの
  callbackやCUDA Navier residualに分散していた処理を削除。
- Host CSR、CUDA CSR、PETScで同じ`TimeIntegrator<Backend>` facadeと時間loopを
  使用。residual、matrix、solver、contextの所有者を呼び出し側に統一。
- `TimeReducedFunctional<Backend>`へ公開APIとadjoint loopを統合。Host CSR、
  CUDA CSR、PETScで`integrator + adjoint matrix + adjoint solver + objective`の
  同じconstructorを使用し、matrix、solver、contextを呼び出し側所有に統一。
- Objectiveの評価方法はstorageの差なので`MemorySpace`で選び、Host CSRと
  PETScは同じHost objective評価、CUDAはDevice objective planを使用。
  Jacobian転置適用、copy、subtract、finalize、同期だけをbackend primitiveへ
  委譲し、objectiveの数式とadjointの逆時間loopはbackendから分離。
- Python時間APIではcontextと作業CSRを小さなbinding ownerに隠し、公開
  constructorを`residual + solver`へ簡略化。
- 定常問題を`Residual<Backend>`、`StateSolver<Backend>`、
  `ReducedFunctional<Backend>`へ移行。Host CSRとPETScが同じforward/adjoint
  制御コードを使い、matrix、solver、contextを呼び出し側所有に統一。
- 定常residualから`Linearization`を除き、state Jacobianのassembleとparameter
  Jacobian転置積を直接contract化。`Linearization.hpp`、分割されていたlinear・
  Newton solverのheader/source、`ReducedFunctional.cpp`を削除。
- Poisson最適化例を新しい定常backend APIへ移行。parameter Jacobian転置積は
  matrix-freeとし、`MatrixLinearization` adapterを不要にした。
- `Operator.hpp`と`LinearOperator` / `MatrixOperator` / `CsrOperator`階層を削除。
  `LinearSolver<Backend>`だけを残し、`OperatorLinearSolver`と`HostLinearSolver`
  aliasも削除。
- ReSolveから旧operator overload、`dynamic_cast<CsrOperator>`、adapter codeを
  削除。Host/Deviceとも具体的な`CsrMatrix`だけを受け取る。
- KSPから`LinearOperator`用MatShell、callback、専用workspaceを削除し、
  `PETScOperator + PetscContext`経路だけに整理。
- `DenseMatrix`と`PETScOperator`のoperator継承を削除。`DenseMatrix`はrow-major
  storageとviewだけに絞り、matvecは既存の`apply` / `applyT`自由関数へ統一。
- 未使用だったfull control Jacobian組立APIを削除。Poisson例、install test、
  solver testを具体CSR/PETSc backend APIへ移行し、旧adapter用testを削除。
- layout取得にしか使われなくなっていた`PETScVector`を削除し、
  `PETScOperator::resize(graph)`がcommunicatorから行分割を直接決定する形に
  簡略化。未使用のPETSc vector assembly/copy API、matrix overload、
  `zeroRowsCols`も削除した。
- PETSc error/MPI検査とHost/PETSc Vec変換を一か所へ統合し、KSPとmatrixで
  重複していた実装を削除した。
- backend固有のsubtract helperを削除し、既存の`axpby`へ統一した。
  `DenseMatrix`と`PETScOperator`の次元APIも`rows()` / `cols()`へ揃えた。
- backend dispatch用の操作接頭辞と`Backend{}`タグ引数を削除した。
  `Backend.hpp`はbackendの型契約だけに限定し、Host/CUDAのvector演算は
  `Vector.hpp` / `Vector.cu`、CSR演算は`CsrMatrix.hpp` / `.cpp` / `.cu`、
  PETSc bridgeは`PETScBackend.hpp`に置く。Host/CUDA/PETScは明示的なcontextを
  受け取る`copy`、`apply`、`applyT`、`axpby`、`finalize`のoverloadで選択し、
  同期はfree functionではなく`ctx.synchronize()`へ統一した。
- `GaussQuadrature`の未使用aliasだった`Quadrature.hpp`を削除した。
- Enzymeは将来のPython APIと`femx-cfd`で使用するため、独立した`femx::ad`
  componentとして維持する。`FEMX_ENABLE_ENZYME`、探索用CMake module、
  install/export、`petsc-enzyme` presetも公開C++ APIの一部として残す。

暫定的に残している境界:

- CUDAの`ConstrainedTimeResidual`実装はdevice mapとkernelを使うため明示特殊化
  のまま。公開contractとcontext所有は共通化済みで、制御順序のtemplate統合は
  inverse移行後の整理対象とする。
- Pythonのtransient backend一覧はHost CSRとReSolveだけを公開する。PETScの
  C++ forward APIは利用可能だが、Python PETScはinverseのbackend化後に戻す。
  C++再設計中はPython API互換を制約にせず、bindingsはC++ API確定後に直す。
- EnzymeのPython公開は、微分対象と配列所有規約を決めてから行う。現時点では
  C++の`femx::ad`を安定した拡張点として保ち、Python互換のためにC++ APIを
  制約しない。`femx-cfd`はinstall/export済みの`femx::ad`を利用する。

次に行う順序:

1. **完了:** `TimeResidual<Backend>`と
   `ConstrainedTimeResidual<Backend>`へ公開APIを移す。
2. **完了:** `TimeReducedFunctional<Backend>`へ公開APIを移し、
   Host/CUDA/PETScのadjoint用matrixとsolverの所有方法を揃える。
3. **完了:** stationary state/inverseを具体backendへ移す。
4. **完了:** 旧operator階層、`AssemblyMatrix`、一時adapter、不要ファイルを
   削除する。

C++側の4段階は完了。次に作業する場合は、この設計を制約せずにPython
bindingsを新APIへ合わせる。

## 残すべきbackend固有コード

次は共通化の対象外とする。

- CUDA kernel本体
- cuBLAS/cuSPARSE API呼び出し
- CUDA descriptor、handle、workspaceのRAII
- PETSc `Mat`/`Vec`/`KSP`とMPI communicator処理
- ReSolveのbind、factorization、transpose cache
- CPUのOpenMP要素ループ
- MPI reduction
- mesh/config reader、VTK writer、Python変換

これらの上にあるアルゴリズムの順序だけを共通化する。

## 完了条件

- `CsrMatrix<Space>`がgraphとvalues以外のbackend状態を持たない。
- ReSolve solverに`dynamic_cast<CsrOperator>`がない。
- `LinearOperator`、`MatrixOperator`、`CsrOperator`が削除されている。
- Host/Device用の`TimeIntegrator`制御コードが一つだけである。
- Host/Device用の`TimeReducedFunctional` adjoint loopが一つだけである。
- Host/Device用の`ConstrainedTimeResidual`制御コードが一つだけである。
- Navier residualのHost/Device差がelement executionと通信だけである。
- Host/Device copyとCUDA同期が明示され、暗黙の転送がない。
- CUDAの時間step内で不要なallocationやdescriptor再作成がない。
- Host、CUDA/ReSolve、PETScの全テストが成功する。
- forward solveとgradientの数値結果がbackend間で許容誤差内に一致する。
- source削減の目安として、重複している制御コードを1,000〜1,600行、
  ファイルを6〜10個削減する。数値目標のために責務を混ぜない。

## 実装上の禁止事項

- `Backend`と`MemorySpace`を同義にしない。
- ReSolveをbackend tagとして直接使わない。ReSolveはCSR backend用solverである。
- `CsrMatrix`にsolver、assembly、CUDA cacheを再導入しない。
- Host/Device両方を持つmirrored containerを導入しない。
- 高水準コードで`if constexpr (Space == ...)`を繰り返さない。
- backendごとに同じ時間ループやadjoint loopを複製しない。
- API互換のためだけに旧operator階層を残さない。
