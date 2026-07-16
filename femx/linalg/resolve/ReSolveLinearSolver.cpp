#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/CsrAssemblyMatrix.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>

#if defined(FEMX_HAS_RESOLVE)
#include <resolve/LinSolverIterative.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/resolve_defs.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>
#if defined(FEMX_RESOLVE_USE_CUDA)
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif
#endif

namespace femx
{
namespace linalg
{

class ReSolveLinearSolver::Impl
{
public:
  explicit Impl(WorkspaceType workspace_type)
    : A_(nullptr),
      workspace_type_(workspace_type)
  {
#if defined(FEMX_HAS_RESOLVE)
    initializeWorkspace();
    resetSolver();
#endif
  }

  Impl(WorkspaceType workspace_type, ReSolveOptions opts)
    : opts_(std::move(opts)),
      A_(nullptr),
      workspace_type_(workspace_type)
  {
#if defined(FEMX_HAS_RESOLVE)
    initializeWorkspace();
    resetSolver();
#endif
  }

  void setOperator(const CsrMatrix& A)
  {
    A_ = &A;

#if defined(FEMX_HAS_RESOLVE)
    const bool reuse_mat =
        mat_ != nullptr
        && mat_rows_ == A.rows()
        && mat_cols_ == A.cols()
        && mat_nnz_ == A.nnz()
        && opts_.factor == "none"
        && opts_.refactor == "none";

    if (!reuse_mat)
    {
      resetSolver();

      mat_ = std::make_unique<ReSolve::matrix::Csr>(
          A.rows(),
          A.cols(),
          A.nnz());
      checkStatus(mat_->allocateAll(memorySpace()),
                  "ReSolve Csr::allocateAll failed");
      mat_rows_ = A.rows();
      mat_cols_ = A.cols();
      mat_nnz_  = A.nnz();
    }

    updateMatrixData(A);

    if (reuse_mat)
    {
      if (opts_.precond != "none")
      {
        checkStatus(solver_->resetPreconditioner(mat_.get()),
                    "ReSolve SystemSolver::resetPreconditioner failed");
      }
    }
    else
    {
      checkStatus(solver_->setMatrix(mat_.get()),
                  "ReSolve SystemSolver::setMatrix failed");
      setupSolver(*solver_, "ReSolve");
    }
#endif
  }

  void setPreconditioner(const std::string& method)
  {
    if (method.empty())
    {
      throw std::runtime_error("Preconditioner method must not be empty");
    }

    opts_.precond = method;

#if defined(FEMX_HAS_RESOLVE)
    resetTransposeOperator();
    if (A_ != nullptr)
    {
      setOperator(*A_);
    }
#endif
  }

  void solve(const Vector<Real>& b, Vector<Real>& x)
  {
    if (A_ == nullptr)
    {
      throw std::runtime_error(
          "LinearSolver::solve() called before setOperator()");
    }
    if (A_->rows() != A_->cols() || b.size() != A_->rows())
    {
      throw std::runtime_error(
          "ReSolveLinearSolver received inconsistent solve dimensions");
    }

    resizeOrZero(x, A_->cols());

#if defined(FEMX_HAS_RESOLVE)
    solveWith(*solver_, b, x, "ReSolve SystemSolver::solve failed");
#else
    throw std::runtime_error(
        "ReSolveLinearSolver was built without ReSolve support");
#endif
  }

  void solve(const LinearOperator& op, const Vector<Real>& rhs, Vector<Real>& out)
  {
    const CsrAssemblyMatrix& sparse_op = requireSparseAssemblyMatrix(op);
    if (op.numRows() != op.numCols() || rhs.size() != op.numRows())
    {
      throw std::runtime_error(
          "ReSolveLinearSolver received inconsistent operator solve dimensions");
    }

    setOperator(sparse_op.mat());
    solve(rhs, out);
  }

  void solveT(const LinearOperator& op, const Vector<Real>& rhs, Vector<Real>& out)
  {
    const CsrAssemblyMatrix& sparse_op = requireSparseAssemblyMatrix(op);
    if (op.numRows() != op.numCols() || rhs.size() != op.numCols())
    {
      throw std::runtime_error(
          "ReSolveLinearSolver received inconsistent transpose solve dimensions");
    }

#if defined(FEMX_HAS_RESOLVE)
    solveTranspose(sparse_op.mat(), rhs, out);
#else
    throw std::runtime_error(
        "ReSolveLinearSolver was built without ReSolve support");
#endif
  }

private:
  struct HostCsrData
  {
    Vector<Index> rp;
    Vector<Index> ci;
    Vector<Real>  vals;
  };

  static const CsrAssemblyMatrix& requireSparseAssemblyMatrix(
      const LinearOperator& op)
  {
    const auto* sparse_op = dynamic_cast<const CsrAssemblyMatrix*>(&op);
    if (sparse_op == nullptr)
    {
      throw std::runtime_error(
          "ReSolveLinearSolver currently supports CsrAssemblyMatrix only");
    }
    return *sparse_op;
  }

  static HostCsrData transposeHost(const CsrMatrix& A)
  {
    HostCsrData data;
    data.rp.assign(A.cols() + 1, 0);
    data.ci.assign(A.nnz(), 0);
    data.vals.assign(A.nnz(), 0.0);

    const Index* rp   = A.rowPtrData();
    const Index* ci   = A.colIndData();
    const Real*  vals = A.valuesData();

    for (Index k = 0; k < A.nnz(); ++k)
    {
      ++data.rp[ci[k] + 1];
    }
    for (Index row = 0; row < A.cols(); ++row)
    {
      data.rp[row + 1] += data.rp[row];
    }

    Vector<Index> next = data.rp;
    for (Index row = 0; row < A.rows(); ++row)
    {
      for (Index k = rp[row]; k < rp[row + 1]; ++k)
      {
        const Index col  = ci[k];
        const Index dest = next[col]++;
        data.ci[dest]    = row;
        data.vals[dest]  = vals[k];
      }
    }

    return data;
  }

  static void checkStatus(int status, const char* msg)
  {
    if (status != 0)
    {
      throw std::runtime_error(msg);
    }
  }

  static void checkStatus(int status, const std::string& msg)
  {
    checkStatus(status, msg.c_str());
  }

#if defined(FEMX_HAS_RESOLVE)
  void solveTranspose(const CsrMatrix& A, const Vector<Real>& b, Vector<Real>& x)
  {
    if (A.rows() != A.cols() || b.size() != A.cols())
    {
      throw std::runtime_error(
          "ReSolveLinearSolver received inconsistent transpose dimensions");
    }
    resizeOrZero(x, A.rows());

    setTransposeOperator(A);
    solveWith(*transpose_solver_,
              b,
              x,
              "ReSolve transpose SystemSolver::solve failed");
  }

  void setTransposeOperator(const CsrMatrix& A)
  {
    transpose_data_ = transposeHost(A);

    const bool reuse_mat =
        mat_t_ != nullptr
        && mat_t_rows_ == A.cols()
        && mat_t_cols_ == A.rows()
        && mat_t_nnz_ == A.nnz()
        && opts_.factor == "none"
        && opts_.refactor == "none";

    if (!reuse_mat)
    {
      resetTransposeSolver();

      mat_t_ = std::make_unique<ReSolve::matrix::Csr>(
          A.cols(),
          A.rows(),
          A.nnz());
      checkStatus(mat_t_->allocateAll(memorySpace()),
                  "ReSolve transpose Csr::allocateAll failed");
      mat_t_rows_ = A.cols();
      mat_t_cols_ = A.rows();
      mat_t_nnz_  = A.nnz();
    }

    updateMatrixData(transpose_data_, *mat_t_);

    if (reuse_mat)
    {
      if (opts_.precond != "none")
      {
        checkStatus(
            transpose_solver_->resetPreconditioner(mat_t_.get()),
            "ReSolve transpose SystemSolver::resetPreconditioner failed");
      }
    }
    else
    {
      checkStatus(transpose_solver_->setMatrix(mat_t_.get()),
                  "ReSolve transpose SystemSolver::setMatrix failed");
      setupSolver(*transpose_solver_, "ReSolve transpose");
    }
  }

  ReSolve::memory::MemorySpace memorySpace() const
  {
    switch (workspace_type_)
    {
    case WorkspaceType::Cpu:
      return ReSolve::memory::HOST;

    case WorkspaceType::Cuda:
      return ReSolve::memory::DEVICE;
    }

    return ReSolve::memory::HOST;
  }

  void updateMatrixData(const CsrMatrix& A)
  {
    if (workspace_type_ == WorkspaceType::Cpu)
    {
      checkStatus(
          mat_->copyFromExternal(A.rowPtrData(),
                                 A.colIndData(),
                                 A.valuesData(),
                                 ReSolve::memory::HOST,
                                 ReSolve::memory::HOST),
          "ReSolve Csr::copyFromExternal to host failed");
      return;
    }

    if (opts_.factor == "klu")
    {
      checkStatus(
          mat_->copyFromExternal(A.rowPtrData(),
                                 A.colIndData(),
                                 A.valuesData(),
                                 ReSolve::memory::HOST,
                                 ReSolve::memory::HOST),
          "ReSolve Csr::copyFromExternal to host failed");
      checkStatus(mat_->syncData(memorySpace()),
                  "ReSolve Csr::syncData failed");
      return;
    }

    checkStatus(
        mat_->copyFromExternal(A.rowPtrData(),
                               A.colIndData(),
                               A.valuesData(),
                               ReSolve::memory::HOST,
                               memorySpace()),
        "ReSolve Csr::copyFromExternal failed");
  }

  void updateMatrixData(const HostCsrData& data, ReSolve::matrix::Csr& A)
  {
    if (workspace_type_ == WorkspaceType::Cpu)
    {
      checkStatus(
          A.copyFromExternal(data.rp.data(),
                             data.ci.data(),
                             data.vals.data(),
                             ReSolve::memory::HOST,
                             ReSolve::memory::HOST),
          "ReSolve transpose Csr::copyFromExternal to host failed");
      return;
    }

    if (opts_.factor == "klu")
    {
      checkStatus(
          A.copyFromExternal(data.rp.data(),
                             data.ci.data(),
                             data.vals.data(),
                             ReSolve::memory::HOST,
                             ReSolve::memory::HOST),
          "ReSolve transpose Csr::copyFromExternal to host failed");
      checkStatus(A.syncData(memorySpace()),
                  "ReSolve transpose Csr::syncData failed");
      return;
    }

    checkStatus(
        A.copyFromExternal(data.rp.data(),
                           data.ci.data(),
                           data.vals.data(),
                           ReSolve::memory::HOST,
                           memorySpace()),
        "ReSolve transpose Csr::copyFromExternal failed");
  }

  void resetSolver()
  {
    solver_ = makeSolver();
    applyOptions(*solver_);
  }

  void resetTransposeSolver()
  {
    transpose_solver_ = makeSolver();
    applyOptions(*transpose_solver_);
  }

  void resetTransposeOperator()
  {
    transpose_solver_.reset();
    mat_t_.reset();
    mat_t_rows_     = 0;
    mat_t_cols_     = 0;
    mat_t_nnz_      = 0;
    transpose_data_ = HostCsrData{};
  }

  std::unique_ptr<ReSolve::SystemSolver> makeSolver()
  {
    switch (workspace_type_)
    {
    case WorkspaceType::Cpu:
      return std::make_unique<ReSolve::SystemSolver>(
          cpu_workspace_.get(),
          opts_.factor,
          opts_.refactor,
          opts_.solve,
          opts_.precond,
          opts_.ir);

    case WorkspaceType::Cuda:
#if defined(FEMX_RESOLVE_USE_CUDA)
      return std::make_unique<ReSolve::SystemSolver>(
          cuda_workspace_.get(),
          opts_.factor,
          opts_.refactor,
          opts_.solve,
          opts_.precond,
          opts_.ir);
#else
      throw std::runtime_error(
          "This ReSolve installation was not built with CUDA support");
#endif
    }

    throw std::runtime_error("Unknown ReSolve workspace type");
  }

  void initializeWorkspace()
  {
    switch (workspace_type_)
    {
    case WorkspaceType::Cpu:
      cpu_workspace_ = std::make_unique<ReSolve::LinAlgWorkspaceCpu>();
      cpu_workspace_->initializeHandles();
      break;

    case WorkspaceType::Cuda:
#if defined(FEMX_RESOLVE_USE_CUDA)
      cuda_workspace_ = std::make_unique<ReSolve::LinAlgWorkspaceCUDA>();
      cuda_workspace_->initializeHandles();
      break;
#else
      throw std::runtime_error(
          "This ReSolve installation was not built with CUDA support");
#endif
    }
  }

  void setupSolver(ReSolve::SystemSolver& solver, const char* prefix)
  {
    if (opts_.factor != "none")
    {
      checkStatus(solver.analyze(),
                  std::string(prefix) + " SystemSolver::analyze failed");
      checkStatus(solver.factorize(),
                  std::string(prefix) + " SystemSolver::factorize failed");
    }

    if (opts_.refactor != "none")
    {
      checkStatus(
          solver.refactorizationSetup(),
          std::string(prefix) + " SystemSolver::refactorizationSetup failed");
      checkStatus(solver.refactorize(),
                  std::string(prefix) + " SystemSolver::refactorize failed");
    }

    if (opts_.precond != "none")
    {
      checkStatus(
          solver.preconditionerSetup(opts_.preconditioner_side),
          std::string(prefix) + " SystemSolver::preconditionerSetup failed");
    }
  }

  void solveWith(ReSolve::SystemSolver& solver,
                 const Vector<Real>&    b,
                 Vector<Real>&          x,
                 const char*            operation)
  {
    const auto memspace = memorySpace();

    ReSolve::vector::Vector rhs(b.size());
    ReSolve::vector::Vector sol(x.size());

    checkStatus(rhs.allocate(memspace),
                "ReSolve rhs Vector<Real>::allocate failed");
    checkStatus(rhs.copyFromExternal(b.data(), ReSolve::memory::HOST, memspace),
                "ReSolve rhs Vector<Real>::copyFromExternal failed");

    checkStatus(sol.allocate(memspace),
                "ReSolve solution Vector<Real>::allocate failed");
    checkStatus(sol.setToZero(memspace),
                "ReSolve solution Vector<Real>::setToZero failed");

    checkStatus(solver.solve(&rhs, &sol), operation);
    // ReSolve may satisfy its Krylov residual estimate before the reported
    // true residual reaches the requested tolerance. Restart from the current
    // solution in that case.
    for (Index correction = 0;
         correction < 2 && needsResidualCorrection(solver);
         ++correction)
    {
      checkStatus(solver.solve(&rhs, &sol), operation);
    }
    checkIterativeConvergence(solver);

    checkStatus(sol.copyToExternal(x.data(),
                                   memspace,
                                   ReSolve::memory::HOST),
                "ReSolve solution Vector<Real>::copyToExternal failed");
  }

  void checkIterativeConvergence(ReSolve::SystemSolver& solver) const
  {
    if (opts_.solve != "fgmres" && opts_.solve != "randgmres")
    {
      return;
    }

    const Real res = solver.getIterativeSolver().getFinalResidualNorm();
    if (!std::isfinite(res) || res > 10.0 * opts_.rtol)
    {
      std::ostringstream msg;
      msg << "ReSolve iterative solve did not converge: final relative "
          << "residual = " << res
          << ", tolerance = " << opts_.rtol
          << ", its = "
          << solver.getIterativeSolver().getNumIter()
          << " / " << opts_.max_its;
      throw std::runtime_error(msg.str());
    }
  }

  bool needsResidualCorrection(ReSolve::SystemSolver& solver) const
  {
    if (opts_.solve != "fgmres" && opts_.solve != "randgmres")
    {
      return false;
    }

    const auto& iterative = solver.getIterativeSolver();
    const Real  res       = iterative.getFinalResidualNorm();
    return std::isfinite(res)
           && res > 10.0 * opts_.rtol
           && iterative.getNumIter() < opts_.max_its;
  }

  void applyOptions(ReSolve::SystemSolver& solver)
  {
    if (opts_.solve == "fgmres" || opts_.solve == "randgmres")
    {
      solver.setGramSchmidtMethod(opts_.gram_schmidt);
      solver.getIterativeSolver().setMaxit(opts_.max_its);
      solver.getIterativeSolver().setTol(opts_.rtol);
      solver.getIterativeSolver().setCliParam("restart", std::to_string(opts_.restart));
      std::string flexible = "no";
      if (opts_.flexible)
      {
        flexible = "yes";
      }
      solver.getIterativeSolver().setCliParam(
          "flexible",
          flexible);

      if (opts_.solve == "randgmres")
      {
        checkStatus(solver.setSketchingMethod(opts_.sketching),
                    "ReSolve SystemSolver::setSketchingMethod failed");
      }
    }
  }
#endif

  ReSolveOptions   opts_;
  const CsrMatrix* A_;
  WorkspaceType    workspace_type_;
  Index            mat_rows_ = 0;
  Index            mat_cols_ = 0;
  Index            mat_nnz_  = 0;
  HostCsrData      transpose_data_;
  Index            mat_t_rows_ = 0;
  Index            mat_t_cols_ = 0;
  Index            mat_t_nnz_  = 0;

#if defined(FEMX_HAS_RESOLVE)
  std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> cpu_workspace_;
#if defined(FEMX_RESOLVE_USE_CUDA)
  std::unique_ptr<ReSolve::LinAlgWorkspaceCUDA> cuda_workspace_;
#endif
  std::unique_ptr<ReSolve::SystemSolver> solver_;
  std::unique_ptr<ReSolve::matrix::Csr>  mat_;
  std::unique_ptr<ReSolve::SystemSolver> transpose_solver_;
  std::unique_ptr<ReSolve::matrix::Csr>  mat_t_;
#endif
};

ReSolveLinearSolver::ReSolveLinearSolver(WorkspaceType workspace_type)
  : impl_(std::make_unique<Impl>(workspace_type))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(WorkspaceType  workspace_type,
                                         ReSolveOptions opts)
  : impl_(std::make_unique<Impl>(workspace_type, std::move(opts)))
{
}

ReSolveLinearSolver::~ReSolveLinearSolver() = default;

void ReSolveLinearSolver::setOperator(const CsrMatrix& A)
{
  impl_->setOperator(A);
}

void ReSolveLinearSolver::solve(const LinearOperator& op,
                                const Vector<Real>&   rhs,
                                Vector<Real>&         out)
{
  impl_->solve(op, rhs, out);
}

void ReSolveLinearSolver::solveT(const LinearOperator& op,
                                 const Vector<Real>&   rhs,
                                 Vector<Real>&         out)
{
  impl_->solveT(op, rhs, out);
}

void ReSolveLinearSolver::setPreconditioner(const std::string& method)
{
  impl_->setPreconditioner(method);
}

void ReSolveLinearSolver::solve(const Vector<Real>& b, Vector<Real>& x)
{
  impl_->solve(b, x);
}

} // namespace linalg
} // namespace femx
