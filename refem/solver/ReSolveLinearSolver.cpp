#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <refem/linalg/SparseMatrix.hpp>
#include <refem/linalg/Vector.hpp>
#include <refem/solver/ReSolveLinearSolver.hpp>

#if defined(REFEM_HAS_RESOLVE)
#include <resolve/LinSolverIterative.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/resolve_defs.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>
#if defined(RESOLVE_USE_CUDA)
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif
#endif

namespace refem
{

class ReSolveLinearSolver::Impl
{
public:
  explicit Impl(WorkspaceType workspace_type)
    : A_(nullptr),
      workspace_type_(workspace_type)
  {
#if defined(REFEM_HAS_RESOLVE)
    initializeWorkspace();
    resetSolver();
#endif
  }

  Impl(WorkspaceType workspace_type, ReSolveOptions options)
    : options_(std::move(options)),
      A_(nullptr),
      workspace_type_(workspace_type)
  {
#if defined(REFEM_HAS_RESOLVE)
    initializeWorkspace();
    resetSolver();
#endif
  }

  void setOperator(const SparseMatrix& A)
  {
    if (A.backend() != MatrixBackend::HostCsr)
    {
      throw std::runtime_error(
          "ReSolveLinearSolver currently accepts HostCsr matrices only");
    }

    A_ = &A;

#if defined(REFEM_HAS_RESOLVE)
    const bool reuse_matrix =
        matrix_ != nullptr && matrix_rows_ == A.rows() && matrix_cols_ == A.cols() && matrix_nnz_ == A.nnz() && options_.factor == "none" && options_.refactor == "none";

    if (!reuse_matrix)
    {
      resetSolver();

      matrix_ = std::make_unique<ReSolve::matrix::Csr>(
          A.rows(),
          A.cols(),
          A.nnz());
      matrix_rows_ = A.rows();
      matrix_cols_ = A.cols();
      matrix_nnz_  = A.nnz();
    }

    updateMatrixData(A);

    if (reuse_matrix)
    {
      if (options_.precond != "none")
      {
        checkStatus(solver_->resetPreconditioner(matrix_.get()),
                    "ReSolve SystemSolver::resetPreconditioner failed");
      }
    }
    else
    {
      checkStatus(solver_->setMatrix(matrix_.get()),
                  "ReSolve SystemSolver::setMatrix failed");

      if (options_.factor != "none")
      {
        checkStatus(solver_->analyze(),
                    "ReSolve SystemSolver::analyze failed");
        checkStatus(solver_->factorize(),
                    "ReSolve SystemSolver::factorize failed");
      }

      if (options_.refactor != "none")
      {
        checkStatus(solver_->refactorizationSetup(),
                    "ReSolve SystemSolver::refactorizationSetup failed");
        checkStatus(solver_->refactorize(),
                    "ReSolve SystemSolver::refactorize failed");
      }

      if (options_.precond != "none")
      {
        checkStatus(solver_->preconditionerSetup(),
                    "ReSolve SystemSolver::preconditionerSetup failed");
      }
    }
#endif
  }

  void setPreconditioner(const std::string& method)
  {
    if (method.empty())
    {
      throw std::runtime_error("Preconditioner method must not be empty");
    }

    options_.precond = method;

#if defined(REFEM_HAS_RESOLVE)
    if (A_ != nullptr)
    {
      setOperator(*A_);
    }
#endif
  }

  void solve(const Vector& b, Vector& x)
  {
    if (A_ == nullptr)
    {
      throw std::runtime_error(
          "LinearSolver::solve() called before setOperator()");
    }

#if defined(REFEM_HAS_RESOLVE)
    const auto memspace = memorySpace();

    ReSolve::vector::Vector rhs(b.size());
    ReSolve::vector::Vector sol(x.size());

    checkStatus(rhs.copyFromExternal(b.data(),
                                     ReSolve::memory::HOST,
                                     memspace),
                "ReSolve rhs Vector::copyFromExternal failed");
    checkStatus(sol.allocate(memspace),
                "ReSolve solution Vector::allocate failed");
    checkStatus(sol.setToZero(memspace),
                "ReSolve solution Vector::setToZero failed");

    checkStatus(solver_->solve(&rhs, &sol),
                "ReSolve SystemSolver::solve failed");

    if (options_.solve == "fgmres" || options_.solve == "randgmres")
    {
      const real_type residual =
          solver_->getIterativeSolver().getFinalResidualNorm();
      if (!std::isfinite(residual) || residual > 10.0 * options_.relative_tolerance)
      {
        std::ostringstream message;
        message << "ReSolve iterative solve did not converge: final relative "
                << "residual = " << residual
                << ", tolerance = " << options_.relative_tolerance
                << ", iterations = "
                << solver_->getIterativeSolver().getNumIter()
                << " / " << options_.max_iterations;
        throw std::runtime_error(message.str());
      }
    }

    checkStatus(sol.copyToExternal(x.data(),
                                   memspace,
                                   ReSolve::memory::HOST),
                "ReSolve solution Vector::copyToExternal failed");
#else
    throw std::runtime_error(
        "ReSolveLinearSolver was built without ReSolve support");
#endif
  }

private:
  static void checkStatus(int status, const char* message)
  {
    if (status != 0)
    {
      throw std::runtime_error(message);
    }
  }

#if defined(REFEM_HAS_RESOLVE)
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

  void updateMatrixData(const SparseMatrix& A)
  {
    if (workspace_type_ == WorkspaceType::Cpu)
    {
      checkStatus(
          matrix_->setDataPointers(
              const_cast<index_type*>(A.rowPtrData()),
              const_cast<index_type*>(A.colIndData()),
              const_cast<real_type*>(A.valuesData()),
              ReSolve::memory::HOST),
          "ReSolve Csr::setDataPointers failed");
      return;
    }

    checkStatus(
        matrix_->copyFromExternal(A.rowPtrData(),
                                  A.colIndData(),
                                  A.valuesData(),
                                  ReSolve::memory::HOST,
                                  memorySpace()),
        "ReSolve Csr::copyFromExternal failed");
  }

  void resetSolver()
  {
    switch (workspace_type_)
    {
    case WorkspaceType::Cpu:
      solver_ = std::make_unique<ReSolve::SystemSolver>(
          cpu_workspace_.get(),
          options_.factor,
          options_.refactor,
          options_.solve,
          options_.precond,
          options_.ir);
      break;

    case WorkspaceType::Cuda:
#if defined(RESOLVE_USE_CUDA)
      solver_ = std::make_unique<ReSolve::SystemSolver>(
          cuda_workspace_.get(),
          options_.factor,
          options_.refactor,
          options_.solve,
          options_.precond,
          options_.ir);
      break;
#else
      throw std::runtime_error(
          "This ReSolve installation was not built with CUDA support");
#endif
    }

    applyOptions();
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
#if defined(RESOLVE_USE_CUDA)
      cuda_workspace_ = std::make_unique<ReSolve::LinAlgWorkspaceCUDA>();
      cuda_workspace_->initializeHandles();
      break;
#else
      throw std::runtime_error(
          "This ReSolve installation was not built with CUDA support");
#endif
    }
  }

  void applyOptions()
  {
    if (options_.solve == "fgmres" || options_.solve == "randgmres")
    {
      solver_->setGramSchmidtMethod(options_.gram_schmidt);
      solver_->getIterativeSolver().setMaxit(options_.max_iterations);
      solver_->getIterativeSolver().setTol(options_.relative_tolerance);
      solver_->getIterativeSolver().setCliParam(
          "restart",
          std::to_string(options_.restart));
      std::string flexible = "no";
      if (options_.flexible)
      {
        flexible = "yes";
      }
      solver_->getIterativeSolver().setCliParam(
          "flexible",
          flexible);

      if (options_.solve == "randgmres")
      {
        checkStatus(solver_->setSketchingMethod(options_.sketching),
                    "ReSolve SystemSolver::setSketchingMethod failed");
      }
    }
  }
#endif

  ReSolveOptions      options_;
  const SparseMatrix* A_;
  WorkspaceType       workspace_type_;
  index_type          matrix_rows_ = 0;
  index_type          matrix_cols_ = 0;
  index_type          matrix_nnz_  = 0;

#if defined(REFEM_HAS_RESOLVE)
  std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> cpu_workspace_;
#if defined(RESOLVE_USE_CUDA)
  std::unique_ptr<ReSolve::LinAlgWorkspaceCUDA> cuda_workspace_;
#endif
  std::unique_ptr<ReSolve::SystemSolver> solver_;
  std::unique_ptr<ReSolve::matrix::Csr>  matrix_;
#endif
};

ReSolveLinearSolver::ReSolveLinearSolver(WorkspaceType workspace_type)
  : impl_(std::make_unique<Impl>(workspace_type))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(WorkspaceType  workspace_type,
                                         ReSolveOptions options)
  : impl_(std::make_unique<Impl>(workspace_type, std::move(options)))
{
}

ReSolveLinearSolver::~ReSolveLinearSolver() = default;

void ReSolveLinearSolver::setOperator(const SparseMatrix& A)
{
  impl_->setOperator(A);
}

void ReSolveLinearSolver::setPreconditioner(const std::string& method)
{
  impl_->setPreconditioner(method);
}

void ReSolveLinearSolver::solve(const Vector& b, Vector& x)
{
  impl_->solve(b, x);
}

} // namespace refem
