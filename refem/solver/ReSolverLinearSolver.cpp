#include <memory>
#include <utility>
#include <stdexcept>
#include <string>

#include <refem/linalg/SparseMatrix.hpp>
#include <refem/linalg/Vector.hpp>
#include <refem/solver/ReSolverLinearSolver.hpp>

#if defined(REFEM_HAS_RESOLVE)
#include <resolve/LinSolverIterative.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/resolve_defs.hpp>
#include <resolve/vector/Vector.hpp>
#endif

namespace refem
{

class ReSolveLinearSolver::Impl
{
public:
  explicit Impl(ReSolve::LinAlgWorkspaceCpu* workspace)
    : A_(nullptr),
      workspace_type_(WorkspaceType::Cpu),
      workspace_(workspace)
  {
#if defined(REFEM_HAS_RESOLVE)
    resetSolver();
#endif
  }

  Impl(ReSolve::LinAlgWorkspaceCpu* workspace, ReSolveOptions options)
    : options_(std::move(options)),
      A_(nullptr),
      workspace_type_(WorkspaceType::Cpu),
      workspace_(workspace)
  {
#if defined(REFEM_HAS_RESOLVE)
    resetSolver();
#endif
  }

  explicit Impl(ReSolve::LinAlgWorkspaceCUDA* workspace)
    : A_(nullptr),
      workspace_type_(WorkspaceType::Cuda),
      workspace_(workspace)
  {
#if defined(REFEM_HAS_RESOLVE)
    resetSolver();
#endif
  }

  Impl(ReSolve::LinAlgWorkspaceCUDA* workspace, ReSolveOptions options)
    : options_(std::move(options)),
      A_(nullptr),
      workspace_type_(WorkspaceType::Cuda),
      workspace_(workspace)
  {
#if defined(REFEM_HAS_RESOLVE)
    resetSolver();
#endif
  }

  explicit Impl(ReSolve::LinAlgWorkspaceHIP* workspace)
    : A_(nullptr),
      workspace_type_(WorkspaceType::Hip),
      workspace_(workspace)
  {
#if defined(REFEM_HAS_RESOLVE)
    resetSolver();
#endif
  }

  Impl(ReSolve::LinAlgWorkspaceHIP* workspace, ReSolveOptions options)
    : options_(std::move(options)),
      A_(nullptr),
      workspace_type_(WorkspaceType::Hip),
      workspace_(workspace)
  {
#if defined(REFEM_HAS_RESOLVE)
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
    resetSolver();

    matrix_ = std::make_unique<ReSolve::matrix::Csr>(
        A.rows(),
        A.cols(),
        A.nnz());

    const auto memspace = memorySpace();

    checkStatus(
        matrix_->copyFromExternal(A.rowPtrData(),
                                  A.colIndData(),
                                  A.valuesData(),
                                  ReSolve::memory::HOST,
                                  memspace),
        "ReSolve Csr::copyFromExternal failed");

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
    case WorkspaceType::Hip:
      return ReSolve::memory::DEVICE;
    }

    return ReSolve::memory::HOST;
  }

  void resetSolver()
  {
    switch (workspace_type_)
    {
    case WorkspaceType::Cpu:
      solver_ = std::make_unique<ReSolve::SystemSolver>(
          static_cast<ReSolve::LinAlgWorkspaceCpu*>(workspace_),
          options_.factor,
          options_.refactor,
          options_.solve,
          options_.precond,
          options_.ir);
      break;

    case WorkspaceType::Cuda:
#if defined(RESOLVE_USE_CUDA)
      solver_ = std::make_unique<ReSolve::SystemSolver>(
          static_cast<ReSolve::LinAlgWorkspaceCUDA*>(workspace_),
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

    case WorkspaceType::Hip:
#if defined(RESOLVE_USE_HIP)
      solver_ = std::make_unique<ReSolve::SystemSolver>(
          static_cast<ReSolve::LinAlgWorkspaceHIP*>(workspace_),
          options_.factor,
          options_.refactor,
          options_.solve,
          options_.precond,
          options_.ir);
      break;
#else
      throw std::runtime_error(
          "This ReSolve installation was not built with HIP support");
#endif
    }

    applyOptions();
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
      solver_->getIterativeSolver().setCliParam(
          "flexible",
          options_.flexible ? "1" : "0");

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
  void*               workspace_;

#if defined(REFEM_HAS_RESOLVE)
  std::unique_ptr<ReSolve::SystemSolver> solver_;
  std::unique_ptr<ReSolve::matrix::Csr>  matrix_;
#endif
};

ReSolveLinearSolver::ReSolveLinearSolver(ReSolve::LinAlgWorkspaceCpu* workspace)
  : impl_(std::make_unique<Impl>(workspace))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(ReSolve::LinAlgWorkspaceCpu* workspace,
                                         ReSolveOptions               options)
  : impl_(std::make_unique<Impl>(workspace, std::move(options)))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(ReSolve::LinAlgWorkspaceCUDA* workspace)
  : impl_(std::make_unique<Impl>(workspace))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(ReSolve::LinAlgWorkspaceCUDA* workspace,
                                         ReSolveOptions                options)
  : impl_(std::make_unique<Impl>(workspace, std::move(options)))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(ReSolve::LinAlgWorkspaceHIP* workspace)
  : impl_(std::make_unique<Impl>(workspace))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(ReSolve::LinAlgWorkspaceHIP* workspace,
                                         ReSolveOptions               options)
  : impl_(std::make_unique<Impl>(workspace, std::move(options)))
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
