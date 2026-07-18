#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <femx/linalg/resolve/ReSolveDeviceSolver.hpp>

#if defined(FEMX_RESOLVE_USE_CUDA)
#include <cublas_v2.h>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/resolve_defs.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif

namespace femx
{
namespace linalg
{

class ReSolveDeviceSolver::Impl
{
public:
  explicit Impl(ReSolveOptions opts)
    : opts_(std::move(opts))
  {
    checkOptions();
#if defined(FEMX_RESOLVE_USE_CUDA)
    static_assert(std::is_same<Index, ReSolve::index_type>::value,
                  "femx/ReSolve index types must match for zero-copy use");
    static_assert(std::is_same<Real, ReSolve::real_type>::value,
                  "femx/ReSolve real types must match for zero-copy use");
    work_ = std::make_unique<ReSolve::LinAlgWorkspaceCUDA>();
    work_->initializeHandles();
#endif
  }

  void setOperator(const DeviceCsrMatrix& mat)
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    bind(fwd_, mat);
#else
    (void) mat;
    unavailable();
#endif
  }

  void solve(const DeviceVector& rhs,
             DeviceVector&       sol,
             CudaContext&        ctx)
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    solveWith(fwd_, rhs, sol, ctx, "ReSolve CUDA solve failed");
#else
    (void) rhs;
    (void) sol;
    (void) ctx;
    unavailable();
#endif
  }

private:
  [[noreturn]] static void unavailable()
  {
    throw std::runtime_error(
        "ReSolveDeviceSolver requires the ReSolve CUDA development backend");
  }

  void checkOptions() const
  {
    if (opts_.solve != "fgmres" && opts_.solve != "randgmres")
    {
      throw std::runtime_error(
          "ReSolveDeviceSolver supports fgmres and randgmres only");
    }
    if (opts_.precond != "ilu0")
    {
      throw std::runtime_error(
          "ReSolveDeviceSolver currently requires the CUDA ilu0 preconditioner");
    }
    if (opts_.factor != "none" || opts_.refactor != "none"
        || opts_.ir != "none")
    {
      throw std::runtime_error(
          "ReSolveDeviceSolver does not stage host direct/refactorization data");
    }
  }

#if defined(FEMX_RESOLVE_USE_CUDA)
  struct System
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

  struct Vecs
  {
    Index                                    size{-1};
    std::unique_ptr<ReSolve::vector::Vector> rhs;
    std::unique_ptr<ReSolve::vector::Vector> sol;
  };

  static void check(int status, const char* op)
  {
    if (status != 0)
    {
      throw std::runtime_error(op);
    }
  }

  void bind(System& sys, const DeviceCsrMatrix& mat)
  {
    if (mat.rows() != mat.cols())
    {
      throw std::runtime_error(
          "ReSolveDeviceSolver requires a square matrix");
    }
    if (mat.rows() <= 0 || mat.nnz() <= 0)
    {
      throw std::runtime_error(
          "ReSolveDeviceSolver requires non-empty device matrix storage");
    }

    const bool same_storage =
        sys.mat != nullptr && sys.solver != nullptr
        && sys.rows == mat.rows() && sys.cols == mat.cols()
        && sys.nnz == mat.nnz() && sys.row_ptr == mat.rowPtrData()
        && sys.col_ind == mat.colIndData() && sys.vals == mat.valsData();

    if (same_storage)
    {
      return;
    }

    sys.solver.reset();
    sys.mat.reset();
    sys.rows           = mat.rows();
    sys.cols           = mat.cols();
    sys.nnz            = mat.nnz();
    sys.row_ptr        = mat.rowPtrData();
    sys.col_ind        = mat.colIndData();
    sys.vals           = const_cast<Real*>(mat.valsData());
    sys.setup_complete = false;
    sys.mat =
        std::make_unique<ReSolve::matrix::Csr>(mat.rows(), mat.cols(), mat.nnz());

    check(sys.mat->setDataPointers(
              const_cast<Index*>(sys.row_ptr),
              const_cast<Index*>(sys.col_ind),
              sys.vals,
              ReSolve::memory::DEVICE),
          "ReSolve Csr::setDataPointers failed");

    sys.solver = makeSolver();
    check(sys.solver->setMatrix(sys.mat.get()),
          "ReSolve SystemSolver::setMatrix failed");
  }

  std::unique_ptr<ReSolve::SystemSolver> makeSolver()
  {
    auto solver = std::make_unique<ReSolve::SystemSolver>(
        work_.get(),
        opts_.factor,
        opts_.refactor,
        opts_.solve,
        opts_.precond,
        opts_.ir);
    applyOpts(*solver);
    return solver;
  }

  void setup(ReSolve::SystemSolver& solver)
  {
    check(solver.preconditionerSetup(opts_.preconditioner_side),
          "ReSolve CUDA preconditioner setup failed");
  }

  void applyOpts(ReSolve::SystemSolver& solver)
  {
    solver.setGramSchmidtMethod(opts_.gram_schmidt);
    auto& iterative = solver.getIterativeSolver();
    iterative.setMaxit(opts_.max_its);
    iterative.setTol(opts_.rtol);
    iterative.setCliParam("restart", std::to_string(opts_.restart));
    iterative.setCliParam("flexible", opts_.flexible ? "yes" : "no");
    if (opts_.solve == "randgmres")
    {
      check(solver.setSketchingMethod(opts_.sketching),
            "ReSolve CUDA sketching setup failed");
    }
  }

  void bindVecs(const DeviceVector& rhs, DeviceVector& sol)
  {
    if (vecs_.size != rhs.size())
    {
      vecs_.rhs  = std::make_unique<ReSolve::vector::Vector>(rhs.size());
      vecs_.sol  = std::make_unique<ReSolve::vector::Vector>(sol.size());
      vecs_.size = rhs.size();
    }

    check(vecs_.rhs->setData(const_cast<Real*>(rhs.data()),
                             ReSolve::memory::DEVICE),
          "ReSolve rhs Vector::setData failed");
    check(vecs_.rhs->setDataUpdated(ReSolve::memory::DEVICE),
          "ReSolve rhs Vector::setDataUpdated failed");
    check(vecs_.sol->setData(sol.data(), ReSolve::memory::DEVICE),
          "ReSolve solution Vector::setData failed");
    check(vecs_.sol->setDataUpdated(ReSolve::memory::DEVICE),
          "ReSolve solution Vector::setDataUpdated failed");
  }

  void solveWith(System&             sys,
                 const DeviceVector& rhs,
                 DeviceVector&       sol,
                 CudaContext&        ctx,
                 const char*         op)
  {
    if (sys.mat == nullptr || sys.solver == nullptr)
    {
      throw std::runtime_error(
          "ReSolveDeviceSolver solve called before binding its matrix");
    }
    if (rhs.size() != sys.rows)
    {
      throw std::runtime_error(
          "ReSolveDeviceSolver RHS has incompatible dimensions");
    }
    if (&rhs == &sol)
    {
      throw std::runtime_error(
          "ReSolveDeviceSolver RHS and solution must differ");
    }
    if (sol.size() != sys.cols)
    {
      sol.resize(sys.cols);
    }
    sol.setZero(ctx);

    // femx assembly owns this stream; ReSolve currently exposes no complete
    // stream hand-off API, so this is the explicit synchronization boundary.
    ctx.synchronize();
    if (rhsIsZero(rhs))
    {
      return;
    }

    check(sys.mat->setUpdated(ReSolve::memory::DEVICE),
          "ReSolve Csr::setUpdated failed");
    if (!sys.setup_complete)
    {
      setup(*sys.solver);
      sys.setup_complete = true;
    }
    else
    {
      check(sys.solver->resetPreconditioner(sys.mat.get()),
            "ReSolve CUDA preconditioner update failed");
    }

    bindVecs(rhs, sol);
    check(sys.solver->solve(vecs_.rhs.get(), vecs_.sol.get()), op);

    // ReSolve currently launches on its own/default stream. Complete that
    // work before the caller resumes using the femx non-blocking stream.
    device::synchronize(nullptr);
  }

  bool rhsIsZero(const DeviceVector& rhs) const
  {
    cublasHandle_t      handle = work_->getCublasHandle();
    cublasPointerMode_t orig_mode{};
    checkCublas(cublasGetPointerMode(handle, &orig_mode),
                "cuBLAS get pointer mode failed");
    checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                "cuBLAS set host pointer mode failed");

    Real                 norm = 0.0;
    const cublasStatus_t norm_status =
        cublasDnrm2(handle, rhs.size(), rhs.data(), 1, &norm);
    const cublasStatus_t restore_status =
        cublasSetPointerMode(handle, orig_mode);
    checkCublas(norm_status, "cuBLAS device RHS norm failed");
    checkCublas(restore_status, "cuBLAS restore pointer mode failed");
    return norm == 0.0;
  }

  static void checkCublas(cublasStatus_t status, const char* op)
  {
    if (status != CUBLAS_STATUS_SUCCESS)
    {
      throw std::runtime_error(op);
    }
  }

  std::unique_ptr<ReSolve::LinAlgWorkspaceCUDA> work_;
  System                                        fwd_;
  Vecs                                          vecs_;
#endif

  ReSolveOptions opts_;
};

ReSolveDeviceSolver::ReSolveDeviceSolver()
  : impl_(std::make_unique<Impl>(ReSolveOptions{}))
{
}

ReSolveDeviceSolver::ReSolveDeviceSolver(ReSolveOptions opts)
  : impl_(std::make_unique<Impl>(std::move(opts)))
{
}

ReSolveDeviceSolver::~ReSolveDeviceSolver() = default;

void ReSolveDeviceSolver::setOperator(const DeviceCsrMatrix& mat)
{
  impl_->setOperator(mat);
}

void ReSolveDeviceSolver::solve(const DeviceVector& rhs,
                                DeviceVector&       sol,
                                CudaContext&        ctx)
{
  impl_->solve(rhs, sol, ctx);
}

} // namespace linalg
} // namespace femx
