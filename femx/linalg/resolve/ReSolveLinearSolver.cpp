#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/native/HostSystemMatrix.hpp>
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
    checkHostMat(mat);
    h_op_ = &mat;

#if defined(FEMX_HAS_RESOLVE)
    ensureCpu();
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

    updateHostMat(mat, *cpu_mat_, "ReSolve Host");
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

  void solve(const HostVector<Real>& rhs, HostVector<Real>& result)
  {
    require(h_op_ != nullptr,
            "ReSolveLinearSolver Host solve called before setOperator");
    require(rhs.size() == h_op_->rows(),
            "ReSolveLinearSolver Host RHS has incompatible dimensions");
    checkHostAliases(*h_op_, rhs, result);

    HostContext ctx;
    auto&       vec_handler = ctx.vectorHandler();
    vec_handler.resizeOrZero(result, h_op_->cols());
    if (isZero(rhs))
    {
      return;
    }

#if defined(FEMX_HAS_RESOLVE)
    solveHostWith(*cpu_solver_,
                  h_vecs_,
                  rhs,
                  result,
                  "ReSolve Host SystemSolver::solve failed");
#else
    unavailableHost();
#endif
  }

  void solve(const HostCsrMatrix&    mat,
             const HostVector<Real>& rhs,
             HostVector<Real>&       result)
  {
    setOperator(mat);
    solve(rhs, result);
  }

  void solveT(const HostCsrMatrix&    mat,
              const HostVector<Real>& rhs,
              HostVector<Real>&       result)
  {
    require(mat.rows() == mat.cols() && rhs.size() == mat.cols(),
            "ReSolveLinearSolver received inconsistent Host transpose dimensions");
    checkHostAliases(mat, rhs, result);
#if defined(FEMX_HAS_RESOLVE)
    solveHostT(mat, rhs, result);
#else
    unavailableHost();
#endif
  }

  void setOperator(const DeviceCsrMatrix& mat)
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    ensureCuda();
    bindCuda(cuda_sys_, mat, *cuda_work_);
#else
    (void) mat;
    unavailableCuda();
#endif
  }

  void solve(const DeviceVector<Real>& rhs,
             DeviceVector<Real>&       result,
             CudaContext&              ctx)
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    require(cuda_work_ != nullptr,
            "ReSolveLinearSolver Device solve called before setOperator");
    solveDeviceWith(cuda_sys_, cuda_vecs_, rhs, result, ctx);
#else
    (void) rhs;
    (void) result;
    (void) ctx;
    unavailableCuda();
#endif
  }

  void solve(const DeviceCsrMatrix&    mat,
             const DeviceVector<Real>& rhs,
             DeviceVector<Real>&       result,
             CudaContext&              ctx)
  {
    setOperator(mat);
    solve(rhs, result, ctx);
  }

  void solveT(const DeviceCsrMatrix&    mat,
              const DeviceVector<Real>& rhs,
              DeviceVector<Real>&       result,
              CudaContext&              ctx)
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    ensureCuda();
    require(mat.rows() == mat.cols() && rhs.size() == mat.cols(),
            "ReSolveLinearSolver received inconsistent Device transpose dimensions");
    CudaSystemMatrix system_mat(ctx);
    system_mat.transpose(mat, d_trans_mat_);
    bindCuda(cuda_trans_system_, d_trans_mat_, *cuda_work_);
    solveDeviceWith(cuda_trans_system_, cuda_vecs_, rhs, result, ctx);
#else
    (void) mat;
    (void) rhs;
    (void) result;
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

  static void checkHostMat(const HostCsrMatrix& mat)
  {
    require(mat.rows() == mat.cols() && mat.rows() > 0 && mat.nnz() > 0
                && mat.vals().size() == mat.nnz(),
            "ReSolveLinearSolver requires a non-empty square Host matrix");
  }

  static void checkHostAliases(const HostCsrMatrix&    mat,
                               const HostVector<Real>& rhs,
                               const HostVector<Real>& result)
  {
    const bool rhs_result = &rhs == &result
                            || (!rhs.empty() && rhs.data() == result.data());
    const bool rhs_mat = &rhs == &mat.vals()
                         || (!rhs.empty() && rhs.data() == mat.valsData());
    const bool result_mat = &result == &mat.vals()
                            || (!result.empty() && result.data() == mat.valsData());
    require(!rhs_result && !rhs_mat && !result_mat,
            "ReSolveLinearSolver Host vectors and matrix values must not alias");
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
  struct HostVecs
  {
    Index                                    size{-1};
    std::unique_ptr<ReSolve::vector::Vector> rhs;
    std::unique_ptr<ReSolve::vector::Vector> result;
  };

  void ensureCpu()
  {
    if (cpu_work_ != nullptr)
    {
      return;
    }
    cpu_work_ = std::make_unique<ReSolve::LinAlgWorkspaceCpu>();
    cpu_work_->initializeHandles();
  }

  std::unique_ptr<ReSolve::SystemSolver> makeCpuSolver()
  {
    auto solver = std::make_unique<ReSolve::SystemSolver>(
        cpu_work_.get(),
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
    ensureCpu();
    cpu_solver_ = makeCpuSolver();
  }

  std::unique_ptr<ReSolve::SystemSolver> makeTransposeSolver()
  {
    auto solver = std::make_unique<ReSolve::SystemSolver>(
        cpu_work_.get(),
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
    ensureCpu();
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

  static void updateHostMat(const HostCsrMatrix&  src,
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

  static void solveHostWith(ReSolve::SystemSolver&  solver,
                            HostVecs&               vecs,
                            const HostVector<Real>& rhs,
                            HostVector<Real>&       result,
                            const char*             op)
  {
    constexpr auto memspace = ReSolve::memory::HOST;
    if (vecs.size != rhs.size())
    {
      vecs.rhs =
          std::make_unique<ReSolve::vector::Vector>(rhs.size());
      vecs.result =
          std::make_unique<ReSolve::vector::Vector>(result.size());
      check(vecs.rhs->allocate(memspace),
            "ReSolve Host rhs Vector::allocate failed");
      check(vecs.result->allocate(memspace),
            "ReSolve Host solution Vector::allocate failed");
      vecs.size = rhs.size();
    }

    check(vecs.rhs->copyFromExternal(
              rhs.data(), ReSolve::memory::HOST, memspace),
          "ReSolve Host rhs Vector::copyFromExternal failed");
    check(vecs.result->setToZero(memspace),
          "ReSolve Host solution Vector::setToZero failed");
    check(solver.solve(vecs.rhs.get(), vecs.result.get()), op);
    check(vecs.result->copyToExternal(
              result.data(), memspace, ReSolve::memory::HOST),
          "ReSolve Host solution Vector::copyToExternal failed");
  }

  void solveHostT(const HostCsrMatrix&    mat,
                  const HostVector<Real>& rhs,
                  HostVector<Real>&       result)
  {
    HostContext ctx;
    auto&       vec_handler = ctx.vectorHandler();
    vec_handler.resizeOrZero(result, mat.rows());
    if (isZero(rhs))
    {
      return;
    }

    setTransposeOperator(mat);
    solveHostWith(*trans_solver_,
                  h_vecs_,
                  rhs,
                  result,
                  "ReSolve transpose SystemSolver::solve failed");
  }

  void setTransposeOperator(const HostCsrMatrix& mat)
  {
    ensureCpu();
    h_mat_.transpose(mat, h_trans_mat_);

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

    updateHostMat(h_trans_mat_, *trans_mat_, "ReSolve transpose");
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
  struct CudaSystem
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

  struct CudaVecs
  {
    Index                                    size{-1};
    std::unique_ptr<ReSolve::vector::Vector> rhs;
    std::unique_ptr<ReSolve::vector::Vector> result;
  };

  void ensureCudaWork(
      std::unique_ptr<ReSolve::LinAlgWorkspaceCUDA>& work)
  {
    if (work != nullptr)
    {
      return;
    }
    checkCudaOpts();
    static_assert(std::is_same<Index, ReSolve::index_type>::value,
                  "femx/ReSolve index types must match for zero-copy use");
    static_assert(std::is_same<Real, ReSolve::real_type>::value,
                  "femx/ReSolve real types must match for zero-copy use");
    work = std::make_unique<ReSolve::LinAlgWorkspaceCUDA>();
    work->initializeHandles();
  }

  void ensureCuda()
  {
    ensureCudaWork(cuda_work_);
  }

  static void resetCudaSystem(CudaSystem& sys)
  {
    sys.solver.reset();
    sys.mat.reset();
    sys.rows           = 0;
    sys.cols           = 0;
    sys.nnz            = 0;
    sys.row_ptr        = nullptr;
    sys.col_ind        = nullptr;
    sys.vals           = nullptr;
    sys.setup_complete = false;
  }

  void bindCuda(CudaSystem&                   sys,
                const DeviceCsrMatrix&        mat,
                ReSolve::LinAlgWorkspaceCUDA& work)
  {
    require(mat.rows() == mat.cols() && mat.rows() > 0 && mat.nnz() > 0
                && mat.vals().size() == mat.nnz(),
            "ReSolveLinearSolver requires a non-empty square Device matrix");

    bindCuda(sys,
             mat.rows(),
             mat.cols(),
             mat.nnz(),
             mat.rowPtrData(),
             mat.colIndData(),
             const_cast<Real*>(mat.valsData()),
             work);
  }

  void bindCuda(CudaSystem&                   sys,
                Index                         rows,
                Index                         cols,
                Index                         nnz,
                const Index*                  row_ptr,
                const Index*                  col_ind,
                Real*                         vals,
                ReSolve::LinAlgWorkspaceCUDA& work)
  {
    require(rows == cols && rows > 0 && nnz > 0 && row_ptr != nullptr
                && col_ind != nullptr && vals != nullptr,
            "ReSolveLinearSolver requires complete square Device CSR storage");

    const bool same_storage =
        sys.mat != nullptr && sys.solver != nullptr
        && sys.rows == rows && sys.cols == cols && sys.nnz == nnz
        && sys.row_ptr == row_ptr && sys.col_ind == col_ind
        && sys.vals == vals;
    if (same_storage)
    {
      return;
    }

    resetCudaSystem(sys);
    sys.rows           = rows;
    sys.cols           = cols;
    sys.nnz            = nnz;
    sys.row_ptr        = row_ptr;
    sys.col_ind        = col_ind;
    sys.vals           = vals;
    sys.setup_complete = false;
    sys.mat            = std::make_unique<ReSolve::matrix::Csr>(
        rows, cols, nnz);
    check(sys.mat->setDataPointers(const_cast<Index*>(sys.row_ptr),
                                   const_cast<Index*>(sys.col_ind),
                                   sys.vals,
                                   ReSolve::memory::DEVICE),
          "ReSolve Device Csr::setDataPointers failed");

    sys.solver = std::make_unique<ReSolve::SystemSolver>(
        &work,
        opts_.factor,
        opts_.refactor,
        opts_.solve,
        opts_.precond,
        opts_.ir);
    applyIterativeOpts(*sys.solver, "ReSolve Device");
    check(sys.solver->setMatrix(sys.mat.get()),
          "ReSolve Device SystemSolver::setMatrix failed");
  }

  static void checkCudaAliases(const CudaSystem&         sys,
                               const DeviceVector<Real>& rhs,
                               const DeviceVector<Real>& result)
  {
    const bool rhs_result = &rhs == &result
                            || (!rhs.empty() && rhs.data() == result.data());
    const bool rhs_mat    = !rhs.empty() && rhs.data() == sys.vals;
    const bool result_mat = !result.empty() && result.data() == sys.vals;
    require(!rhs_result && !rhs_mat && !result_mat,
            "ReSolveLinearSolver Device vectors and matrix values must not alias");
  }

  void bindCudaVecs(CudaVecs&                 vecs,
                    const DeviceVector<Real>& rhs,
                    DeviceVector<Real>&       result)
  {
    if (vecs.size != rhs.size())
    {
      vecs.rhs    = std::make_unique<ReSolve::vector::Vector>(rhs.size());
      vecs.result = std::make_unique<ReSolve::vector::Vector>(result.size());
      vecs.size   = rhs.size();
    }

    check(vecs.rhs->setData(const_cast<Real*>(rhs.data()),
                            ReSolve::memory::DEVICE),
          "ReSolve Device rhs Vector::setData failed");
    check(vecs.rhs->setDataUpdated(ReSolve::memory::DEVICE),
          "ReSolve Device rhs Vector::setDataUpdated failed");
    check(vecs.result->setData(result.data(), ReSolve::memory::DEVICE),
          "ReSolve Device solution Vector::setData failed");
    check(vecs.result->setDataUpdated(ReSolve::memory::DEVICE),
          "ReSolve Device solution Vector::setDataUpdated failed");
  }

  void solveDeviceWith(CudaSystem&               sys,
                       CudaVecs&                 vecs,
                       const DeviceVector<Real>& rhs,
                       DeviceVector<Real>&       result,
                       CudaContext&              ctx)
  {
    require(sys.mat != nullptr && sys.solver != nullptr,
            "ReSolveLinearSolver Device solve called before setOperator");
    require(rhs.size() == sys.rows,
            "ReSolveLinearSolver Device RHS has incompatible dimensions");
    checkCudaAliases(sys, rhs, result);
    auto& vec_handler = ctx.vectorHandler();
    vec_handler.resizeOrZero(result, sys.cols);

    // femx assembly owns this stream. ReSolve currently has no complete stream
    // hand-off API, so this is the explicit producer/solver boundary.
    ctx.sync();

    check(sys.mat->setUpdated(ReSolve::memory::DEVICE),
          "ReSolve Device Csr::setUpdated failed");
    if (!sys.setup_complete)
    {
      check(sys.solver->preconditionerSetup(opts_.pc_side),
            "ReSolve Device preconditioner setup failed");
      sys.setup_complete = true;
    }
    else
    {
      check(sys.solver->resetPreconditioner(sys.mat.get()),
            "ReSolve Device preconditioner update failed");
    }

    bindCudaVecs(vecs, rhs, result);
    check(sys.solver->solve(vecs.rhs.get(), vecs.result.get()),
          "ReSolve Device solve failed");

    // ReSolve currently launches on its own/default stream. Complete it before
    // the caller resumes work on the femx non-blocking stream.
    cuda::sync(nullptr);
  }

#endif

  ReSolveOptions       opts_;
  HostContext          h_matrix_ctx_;
  HostSystemMatrix     h_mat_{h_matrix_ctx_};
  const HostCsrMatrix* h_op_{nullptr};
  Index                cpu_rows_{0};
  Index                cpu_cols_{0};
  Index                cpu_nnz_{0};
  HostCsrMatrix        h_trans_mat_;
  Index                trans_rows_{0};
  Index                trans_cols_{0};
  Index                trans_nnz_{0};

#if defined(FEMX_HAS_RESOLVE)
  std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> cpu_work_;
  std::unique_ptr<ReSolve::SystemSolver>       cpu_solver_;
  std::unique_ptr<ReSolve::matrix::Csr>        cpu_mat_;
  std::unique_ptr<ReSolve::SystemSolver>       trans_solver_;
  std::unique_ptr<ReSolve::matrix::Csr>        trans_mat_;
  HostVecs                                     h_vecs_;
#endif

#if defined(FEMX_RESOLVE_USE_CUDA)
  std::unique_ptr<ReSolve::LinAlgWorkspaceCUDA> cuda_work_;
  CudaSystem                                    cuda_sys_;
  CudaVecs                                      cuda_vecs_;
  DeviceCsrMatrix                               d_trans_mat_;
  CudaSystem                                    cuda_trans_system_;
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
                                HostVector<Real>&       result,
                                Context<MemorySpace::Host>&)
{
  impl_->solve(mat, rhs, result);
}

void ReSolveLinearSolver::solveT(const HostCsrMatrix&    mat,
                                 const HostVector<Real>& rhs,
                                 HostVector<Real>&       result,
                                 Context<MemorySpace::Host>&)
{
  impl_->solveT(mat, rhs, result);
}

void ReSolveLinearSolver::solve(const DeviceCsrMatrix&        mat,
                                const DeviceVector<Real>&     rhs,
                                DeviceVector<Real>&           result,
                                Context<MemorySpace::Device>& ctx)
{
  impl_->solve(mat, rhs, result, dynamic_cast<CudaContext&>(ctx));
}

void ReSolveLinearSolver::solveT(const DeviceCsrMatrix&        mat,
                                 const DeviceVector<Real>&     rhs,
                                 DeviceVector<Real>&           result,
                                 Context<MemorySpace::Device>& ctx)
{
  impl_->solveT(mat, rhs, result, dynamic_cast<CudaContext&>(ctx));
}

} // namespace linalg
} // namespace femx
