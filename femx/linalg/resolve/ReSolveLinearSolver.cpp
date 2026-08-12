#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/common/Device.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/host/HostContext.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>

#if defined(FEMX_HAS_RESOLVE)
#include <resolve/LinSolverIterative.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/resolve_defs.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>
#endif

#if defined(FEMX_RESOLVE_USE_CUDA)
#include <resolve/LinSolverDirect.hpp>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif

namespace femx
{
namespace linalg
{

class ReSolveLinearSolver::Impl
{
public:
  explicit Impl(ReSolveOptions opts)
    : opts_(std::move(opts))
  {
    checkOpts();
  }

  void setOperator(const HostCsrMatrix& mat)
  {
    checkHostMatrix(mat);
    h_op_ = &mat;

#if defined(FEMX_HAS_RESOLVE)
    initializeCpuWorkspace();
    const bool reuse = cpu_mat_ != nullptr
                       && cpu_rows_ == mat.rows()
                       && cpu_cols_ == mat.cols()
                       && cpu_nnz_ == mat.nnz()
                       && opts_.factor == "none"
                       && opts_.refactor == "none";

    if (!reuse)
    {
      resetCpuSolver();
      cpu_mat_ = std::make_unique<ReSolve::matrix::Csr>(
          mat.rows(), mat.cols(), mat.nnz());
      check(cpu_mat_->allocateAll(ReSolve::memory::HOST),
            "ReSolve Host Csr::allocateAll failed");
      cpu_rows_ = mat.rows();
      cpu_cols_ = mat.cols();
      cpu_nnz_  = mat.nnz();
    }

    setHostMatrix(mat, *cpu_mat_, "ReSolve Host");
    if (reuse)
    {
      if (opts_.precond != "none")
      {
        check(cpu_solver_->resetPreconditioner(cpu_mat_.get()),
              "ReSolve Host preconditioner update failed");
      }
    }
    else
    {
      check(cpu_solver_->setMatrix(cpu_mat_.get()),
            "ReSolve Host SystemSolver::setMatrix failed");
      setupCpu(*cpu_solver_, "ReSolve Host");
    }
#endif
  }

  void solve(const HostVector<Real>& rhs, HostVector<Real>& x)
  {
    require(h_op_ != nullptr,
            "ReSolveLinearSolver Host solve called before setOperator");
    require(rhs.size() == h_op_->rows(),
            "ReSolveLinearSolver Host RHS has incompatible dimensions");
    checkHostBuffers(*h_op_, rhs, x);

    HostContext ctx;
    auto&       vec_handler = ctx.vectorHandler();
    vec_handler.assign(x, h_op_->cols(), 0);
    if (isZero(rhs))
    {
      return;
    }

#if defined(FEMX_HAS_RESOLVE)
    solveHost(*cpu_solver_,
              host_vectors_,
              rhs,
              x,
              "ReSolve Host SystemSolver::solve failed");
#else
    unavailableHost();
#endif
  }

  void solve(const HostCsrMatrix&    mat,
             const HostVector<Real>& rhs,
             HostVector<Real>&       x)
  {
    setOperator(mat);
    solve(rhs, x);
  }

  void solveT(const HostCsrMatrix&    mat,
              const HostVector<Real>& rhs,
              HostVector<Real>&       x)
  {
    require(mat.rows() == mat.cols() && rhs.size() == mat.cols(),
            "ReSolveLinearSolver received inconsistent Host transpose dimensions");
    checkHostBuffers(mat, rhs, x);

#if defined(FEMX_HAS_RESOLVE)
    solveHostT(mat, rhs, x);
#else
    unavailableHost();
#endif
  }

  void setOperator(const DeviceCsrMatrix& mat)
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    initializeCudaWorkspace();
    setCudaMatrix(cuda_state_, mat, *cuda_workspace_);
#else
    (void) mat;
    unavailableCuda();
#endif
  }

  void solve(const DeviceVector<Real>& rhs,
             DeviceVector<Real>&       x,
             CudaContext&              ctx)
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    require(cuda_workspace_ != nullptr,
            "ReSolveLinearSolver Device solve called before setOperator");
    solveCuda(cuda_state_, cuda_vectors_, rhs, x, ctx);
#else
    (void) rhs;
    (void) x;
    (void) ctx;
    unavailableCuda();
#endif
  }

  void solve(const DeviceCsrMatrix&    mat,
             const DeviceVector<Real>& rhs,
             DeviceVector<Real>&       x,
             CudaContext&              ctx)
  {
    setOperator(mat);
    solve(rhs, x, ctx);
  }

  void solveT(const DeviceCsrMatrix&    mat,
              const DeviceVector<Real>& rhs,
              DeviceVector<Real>&       x,
              CudaContext&              ctx)
  {
#if defined(FEMX_RESOLVE_USE_CUDA)

    initializeCudaWorkspace();
    require(mat.rows() == mat.cols() && rhs.size() == mat.cols(),
            "ReSolveLinearSolver received inconsistent Device transpose dimensions");
    ctx.matrixHandler().transpose(mat, d_trans_mat_);
    setCudaMatrix(cuda_transpose_state_, d_trans_mat_, *cuda_workspace_);
    solveCuda(cuda_transpose_state_, cuda_vectors_, rhs, x, ctx);

#else

    (void) mat;
    (void) rhs;
    (void) x;
    (void) ctx;
    unavailableCuda();

#endif
  }

private:
  void checkOpts() const
  {
    require(opts_.max_its > 0 && opts_.restart > 0
                && std::isfinite(opts_.rtol) && opts_.rtol > 0.0,
            "ReSolveLinearSolver iteration options must be positive and finite");
    require(opts_.pc_side == "left" || opts_.pc_side == "right",
            "ReSolveLinearSolver preconditioner side must be left or right");
  }

  void checkCudaOpts() const
  {
    require(opts_.solve == "fgmres" || opts_.solve == "randgmres",
            "ReSolveLinearSolver Device path supports fgmres and randgmres only");
    require(opts_.precond == "ilu0",
            "ReSolveLinearSolver Device path requires the ilu0 preconditioner");
    require(opts_.factor == "none" && opts_.refactor == "none"
                && opts_.ir == "none",
            "ReSolveLinearSolver Device path does not stage Host direct data");
  }

  static void checkHostMatrix(const HostCsrMatrix& mat)
  {
    require(mat.rows() == mat.cols() && mat.rows() > 0 && mat.nnz() > 0
                && mat.vals().size() == mat.nnz(),
            "ReSolveLinearSolver requires a non-empty square Host matrix");
  }

  static void checkHostBuffers(const HostCsrMatrix&    mat,
                               const HostVector<Real>& rhs,
                               const HostVector<Real>& x)
  {
    const bool rhs_x   = &rhs == &x || (!rhs.empty() && rhs.data() == x.data());
    const bool rhs_mat = &rhs == &mat.vals() || (!rhs.empty() && rhs.data() == mat.valsData());
    const bool x_mat   = &x == &mat.vals() || (!x.empty() && x.data() == mat.valsData());

    require(!rhs_x && !rhs_mat && !x_mat,
            "ReSolveLinearSolver Host vectors and matrix values must use separate storage");
  }

  static bool isZero(const HostVector<Real>& vals)
  {
    return std::all_of(vals.begin(),
                       vals.end(),
                       [](Real val)
                       {
                         return val == 0.0;
                       });
  }

  [[noreturn]] static void unavailableHost()
  {
    throw std::runtime_error(
        "ReSolveLinearSolver was built without ReSolve Host support");
  }

  [[noreturn]] static void unavailableCuda()
  {
    throw std::runtime_error(
        "ReSolveLinearSolver Device path requires ReSolve CUDA support");
  }

  static void check(int status, const char* op)
  {
    if (status != 0)
    {
      throw std::runtime_error(op);
    }
  }

  static void check(int status, const std::string& op)
  {
    check(status, op.c_str());
  }

#if defined(FEMX_HAS_RESOLVE)
  struct HostVectors
  {
    Index                                    size{-1};
    std::unique_ptr<ReSolve::vector::Vector> rhs;
    std::unique_ptr<ReSolve::vector::Vector> x;
  };

  void initializeCpuWorkspace()
  {
    if (cpu_workspace_ != nullptr)
    {
      return;
    }
    cpu_workspace_ = std::make_unique<ReSolve::LinAlgWorkspaceCpu>();
    cpu_workspace_->initializeHandles();
  }

  std::unique_ptr<ReSolve::SystemSolver> makeCpuSolver()
  {
    auto solver = std::make_unique<ReSolve::SystemSolver>(
        cpu_workspace_.get(),
        opts_.factor,
        opts_.refactor,
        opts_.solve,
        opts_.precond,
        opts_.ir);
    applyIterativeOpts(*solver, "ReSolve Host");
    return solver;
  }

  void resetCpuSolver()
  {
    initializeCpuWorkspace();
    cpu_solver_ = makeCpuSolver();
  }

  std::unique_ptr<ReSolve::SystemSolver> makeTransposeSolver()
  {
    auto solver = std::make_unique<ReSolve::SystemSolver>(
        cpu_workspace_.get(),
        opts_.factor,
        opts_.refactor,
        opts_.solve,
        opts_.precond,
        opts_.ir);
    applyIterativeOpts(*solver, "ReSolve transpose");
    return solver;
  }

  void resetTransposeSolver()
  {
    initializeCpuWorkspace();
    trans_solver_ = makeTransposeSolver();
  }

  void applyIterativeOpts(ReSolve::SystemSolver& solver,
                          const char*            prefix)
  {
    if (opts_.solve != "fgmres" && opts_.solve != "randgmres")
    {
      return;
    }

    solver.setGramSchmidtMethod(opts_.gram_schmidt);
    auto& iterative = solver.getIterativeSolver();
    iterative.setMaxit(opts_.max_its);
    iterative.setTol(opts_.rtol);
    iterative.setCliParam("restart", std::to_string(opts_.restart));
    iterative.setCliParam("flexible", opts_.flexible ? "yes" : "no");
    if (opts_.solve == "randgmres")
    {
      check(solver.setSketchingMethod(opts_.sketching),
            std::string(prefix) + " sketching setup failed");
    }
  }

  void setupCpu(ReSolve::SystemSolver& solver, const char* prefix)
  {
    if (opts_.factor != "none")
    {
      check(solver.analyze(),
            std::string(prefix) + " SystemSolver::analyze failed");
      check(solver.factorize(),
            std::string(prefix) + " SystemSolver::factorize failed");
    }

    if (opts_.refactor != "none")
    {
      check(solver.refactorizationSetup(),
            std::string(prefix) + " refactorizationSetup failed");
      check(solver.refactorize(),
            std::string(prefix) + " SystemSolver::refactorize failed");
    }

    if (opts_.precond != "none")
    {
      check(solver.preconditionerSetup(opts_.pc_side),
            std::string(prefix) + " preconditioner setup failed");
    }
  }

  static void setHostMatrix(const HostCsrMatrix&  src,
                            ReSolve::matrix::Csr& dst,
                            const char*           prefix)
  {
    check(dst.copyFromExternal(src.rowPtrData(),
                               src.colIndData(),
                               src.valsData(),
                               ReSolve::memory::HOST,
                               ReSolve::memory::HOST),
          std::string(prefix) + " Csr::copyFromExternal failed");
  }

  static void solveHost(ReSolve::SystemSolver&  solver,
                        HostVectors&            vectors,
                        const HostVector<Real>& rhs,
                        HostVector<Real>&       x,
                        const char*             op)
  {
    constexpr auto memspace = ReSolve::memory::HOST;
    if (vectors.size != rhs.size())
    {
      vectors.rhs =
          std::make_unique<ReSolve::vector::Vector>(rhs.size());
      vectors.x = std::make_unique<ReSolve::vector::Vector>(x.size());
      check(vectors.rhs->allocate(memspace),
            "ReSolve Host rhs Vector::allocate failed");
      check(vectors.x->allocate(memspace),
            "ReSolve Host solution Vector::allocate failed");
      vectors.size = rhs.size();
    }

    check(vectors.rhs->copyFromExternal(
              rhs.data(), ReSolve::memory::HOST, memspace),
          "ReSolve Host rhs Vector::copyFromExternal failed");
    check(vectors.x->setToZero(memspace),
          "ReSolve Host solution Vector::setToZero failed");
    check(solver.solve(vectors.rhs.get(), vectors.x.get()), op);
    check(vectors.x->copyToExternal(
              x.data(), memspace, ReSolve::memory::HOST),
          "ReSolve Host solution Vector::copyToExternal failed");
  }

  void solveHostT(const HostCsrMatrix&    mat,
                  const HostVector<Real>& rhs,
                  HostVector<Real>&       x)
  {
    HostContext ctx;
    auto&       vec_handler = ctx.vectorHandler();
    vec_handler.assign(x, mat.rows(), 0);
    if (isZero(rhs))
    {
      return;
    }

    setTransposeOperator(mat);
    solveHost(*trans_solver_,
              host_vectors_,
              rhs,
              x,
              "ReSolve transpose SystemSolver::solve failed");
  }

  void setTransposeOperator(const HostCsrMatrix& mat)
  {
    initializeCpuWorkspace();
    h_mat_ctx_.matrixHandler().transpose(mat, h_trans_mat_);

    const bool reuse = trans_mat_ != nullptr
                       && trans_rows_ == mat.cols()
                       && trans_cols_ == mat.rows()
                       && trans_nnz_ == mat.nnz()
                       && opts_.factor == "none"
                       && opts_.refactor == "none";
    if (!reuse)
    {
      resetTransposeSolver();
      trans_mat_ = std::make_unique<ReSolve::matrix::Csr>(
          mat.cols(), mat.rows(), mat.nnz());
      check(trans_mat_->allocateAll(ReSolve::memory::HOST),
            "ReSolve transpose Csr::allocateAll failed");
      trans_rows_ = mat.cols();
      trans_cols_ = mat.rows();
      trans_nnz_  = mat.nnz();
    }

    setHostMatrix(h_trans_mat_, *trans_mat_, "ReSolve transpose");
    if (reuse)
    {
      if (opts_.precond != "none")
      {
        check(trans_solver_->resetPreconditioner(trans_mat_.get()),
              "ReSolve transpose preconditioner update failed");
      }
    }
    else
    {
      check(trans_solver_->setMatrix(trans_mat_.get()),
            "ReSolve transpose SystemSolver::setMatrix failed");
      setupCpu(*trans_solver_, "ReSolve transpose");
    }
  }
#endif

#if defined(FEMX_RESOLVE_USE_CUDA)
  struct CudaSolverState
  {
    std::unique_ptr<ReSolve::matrix::Csr>  mat;
    std::unique_ptr<ReSolve::SystemSolver> solver;
    Index                                  rows{0};
    Index                                  cols{0};
    Index                                  nnz{0};
    const Index*                           row_ptr{nullptr};
    const Index*                           col_ind{nullptr};
    Real*                                  vals{nullptr};
    bool                                   setup_complete{false};
  };

  struct CudaVectors
  {
    Index                                    size{-1};
    std::unique_ptr<ReSolve::vector::Vector> rhs;
    std::unique_ptr<ReSolve::vector::Vector> x;
  };

  void initializeCudaWorkspace()
  {
    if (cuda_workspace_ != nullptr)
    {
      return;
    }
    checkCudaOpts();
    static_assert(std::is_same<Index, ReSolve::index_type>::value,
                  "femx/ReSolve index types must match for zero-copy use");
    static_assert(std::is_same<Real, ReSolve::real_type>::value,
                  "femx/ReSolve real types must match for zero-copy use");

    cuda_workspace_ = std::make_unique<ReSolve::LinAlgWorkspaceCUDA>();
    cuda_workspace_->initializeHandles();
  }

  static void resetCudaState(CudaSolverState& state)
  {
    state.solver.reset();
    state.mat.reset();
    state.rows           = 0;
    state.cols           = 0;
    state.nnz            = 0;
    state.row_ptr        = nullptr;
    state.col_ind        = nullptr;
    state.vals           = nullptr;
    state.setup_complete = false;
  }

  void setCudaMatrix(CudaSolverState&              state,
                     const DeviceCsrMatrix&        mat,
                     ReSolve::LinAlgWorkspaceCUDA& workspace)
  {
    require(mat.rows() == mat.cols() && mat.rows() > 0 && mat.nnz() > 0
                && mat.vals().size() == mat.nnz(),
            "ReSolveLinearSolver requires a non-empty square Device matrix");

    setCudaMatrix(state,
                  mat.rows(),
                  mat.cols(),
                  mat.nnz(),
                  mat.rowPtrData(),
                  mat.colIndData(),
                  const_cast<Real*>(mat.valsData()),
                  workspace);
  }

  void setCudaMatrix(CudaSolverState&              state,
                     Index                         rows,
                     Index                         cols,
                     Index                         nnz,
                     const Index*                  row_ptr,
                     const Index*                  col_ind,
                     Real*                         vals,
                     ReSolve::LinAlgWorkspaceCUDA& workspace)
  {
    require(rows == cols && rows > 0 && nnz > 0 && row_ptr != nullptr
                && col_ind != nullptr && vals != nullptr,
            "ReSolveLinearSolver requires complete square Device CSR storage");

    const bool same_storage =
        state.mat != nullptr && state.solver != nullptr
        && state.rows == rows && state.cols == cols && state.nnz == nnz
        && state.row_ptr == row_ptr && state.col_ind == col_ind
        && state.vals == vals;
    if (same_storage)
    {
      return;
    }

    resetCudaState(state);
    state.rows           = rows;
    state.cols           = cols;
    state.nnz            = nnz;
    state.row_ptr        = row_ptr;
    state.col_ind        = col_ind;
    state.vals           = vals;
    state.setup_complete = false;

    state.mat = std::make_unique<ReSolve::matrix::Csr>(rows, cols, nnz);
    check(state.mat->setDataPointers(const_cast<Index*>(state.row_ptr),
                                     const_cast<Index*>(state.col_ind),
                                     state.vals,
                                     ReSolve::memory::DEVICE),
          "ReSolve Device Csr::setDataPointers failed");

    state.solver = std::make_unique<ReSolve::SystemSolver>(
        &workspace,
        opts_.factor,
        opts_.refactor,
        opts_.solve,
        opts_.precond,
        opts_.ir);

    check(state.solver->getPreconditionerSolver().setCliParam(
              "numeric_boost", "no"),
          "ReSolve Device ILU0 numeric boost setup failed");
    applyIterativeOpts(*state.solver, "ReSolve Device");
    check(state.solver->setMatrix(state.mat.get()),
          "ReSolve Device SystemSolver::setMatrix failed");
  }

  static void checkCudaBuffers(const CudaSolverState&    state,
                               const DeviceVector<Real>& rhs,
                               const DeviceVector<Real>& x)
  {
    const bool rhs_x   = &rhs == &x || (!rhs.empty() && rhs.data() == x.data());
    const bool rhs_mat = !rhs.empty() && rhs.data() == state.vals;
    const bool x_mat   = !x.empty() && x.data() == state.vals;
    require(!rhs_x && !rhs_mat && !x_mat,
            "ReSolveLinearSolver Device vectors and matrix values must use separate storage");
  }

  void setCudaVectors(CudaVectors&              vectors,
                      const DeviceVector<Real>& rhs,
                      DeviceVector<Real>&       x)
  {
    if (vectors.size != rhs.size())
    {
      vectors.rhs  = std::make_unique<ReSolve::vector::Vector>(rhs.size());
      vectors.x    = std::make_unique<ReSolve::vector::Vector>(x.size());
      vectors.size = rhs.size();
    }

    check(vectors.rhs->setData(const_cast<Real*>(rhs.data()),
                               ReSolve::memory::DEVICE),
          "ReSolve Device rhs Vector::setData failed");
    check(vectors.rhs->setDataUpdated(ReSolve::memory::DEVICE),
          "ReSolve Device rhs Vector::setDataUpdated failed");
    check(vectors.x->setData(x.data(), ReSolve::memory::DEVICE),
          "ReSolve Device solution Vector::setData failed");
    check(vectors.x->setDataUpdated(ReSolve::memory::DEVICE),
          "ReSolve Device solution Vector::setDataUpdated failed");
  }

  void solveCuda(CudaSolverState&          state,
                 CudaVectors&              vectors,
                 const DeviceVector<Real>& rhs,
                 DeviceVector<Real>&       x,
                 CudaContext&              ctx)
  {
    require(state.mat != nullptr && state.solver != nullptr,
            "ReSolveLinearSolver Device solve called before setOperator");
    require(rhs.size() == state.rows,
            "ReSolveLinearSolver Device RHS has incompatible dimensions");
    checkCudaBuffers(state, rhs, x);
    auto& vec_handler = ctx.vectorHandler();
    vec_handler.assign(x, state.cols, 0);

    // femx assembly owns this stream. ReSolve currently has no complete stream
    // hand-off API, so this is the explicit producer/solver boundary.
    ctx.sync();

    check(state.mat->setUpdated(ReSolve::memory::DEVICE),
          "ReSolve Device Csr::setUpdated failed");
    if (!state.setup_complete)
    {
      check(state.solver->preconditionerSetup(opts_.pc_side),
            "ReSolve Device preconditioner setup failed");
      state.setup_complete = true;
    }
    else
    {
      check(state.solver->resetPreconditioner(state.mat.get()),
            "ReSolve Device preconditioner update failed");
    }

    setCudaVectors(vectors, rhs, x);
    check(state.solver->solve(vectors.rhs.get(), vectors.x.get()),
          "ReSolve Device solve failed");

    // ReSolve currently launches on its own/default stream. Complete it before
    // the caller resumes work on the femx non-blocking stream.
    device::sync(nullptr);
  }

#endif

  ReSolveOptions       opts_;
  HostContext          h_mat_ctx_;
  const HostCsrMatrix* h_op_{nullptr};
  Index                cpu_rows_{0};
  Index                cpu_cols_{0};
  Index                cpu_nnz_{0};
  HostCsrMatrix        h_trans_mat_;
  Index                trans_rows_{0};
  Index                trans_cols_{0};
  Index                trans_nnz_{0};

#if defined(FEMX_HAS_RESOLVE)
  std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> cpu_workspace_;
  std::unique_ptr<ReSolve::SystemSolver>       cpu_solver_;
  std::unique_ptr<ReSolve::matrix::Csr>        cpu_mat_;
  std::unique_ptr<ReSolve::SystemSolver>       trans_solver_;
  std::unique_ptr<ReSolve::matrix::Csr>        trans_mat_;
  HostVectors                                  host_vectors_;
#endif

#if defined(FEMX_RESOLVE_USE_CUDA)
  std::unique_ptr<ReSolve::LinAlgWorkspaceCUDA> cuda_workspace_;
  CudaSolverState                               cuda_state_;
  CudaVectors                                   cuda_vectors_;
  DeviceCsrMatrix                               d_trans_mat_;
  CudaSolverState                               cuda_transpose_state_;
#endif
};

ReSolveLinearSolver::ReSolveLinearSolver()
  : impl_(std::make_unique<Impl>(ReSolveOptions{}))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(ReSolveOptions opts)
  : impl_(std::make_unique<Impl>(std::move(opts)))
{
}

ReSolveLinearSolver::~ReSolveLinearSolver() = default;

void ReSolveLinearSolver::solve(const HostCsrMatrix&    mat,
                                const HostVector<Real>& rhs,
                                HostVector<Real>&       x,
                                Context<MemorySpace::Host>&)
{
  impl_->solve(mat, rhs, x);
}

void ReSolveLinearSolver::solveT(const HostCsrMatrix&    mat,
                                 const HostVector<Real>& rhs,
                                 HostVector<Real>&       x,
                                 Context<MemorySpace::Host>&)
{
  impl_->solveT(mat, rhs, x);
}

void ReSolveLinearSolver::solve(const DeviceCsrMatrix&        mat,
                                const DeviceVector<Real>&     rhs,
                                DeviceVector<Real>&           x,
                                Context<MemorySpace::Device>& ctx)
{
  impl_->solve(mat, rhs, x, dynamic_cast<CudaContext&>(ctx));
}

void ReSolveLinearSolver::solveT(const DeviceCsrMatrix&        mat,
                                 const DeviceVector<Real>&     rhs,
                                 DeviceVector<Real>&           x,
                                 Context<MemorySpace::Device>& ctx)
{
  impl_->solveT(mat, rhs, x, dynamic_cast<CudaContext&>(ctx));
}

} // namespace linalg
} // namespace femx
