#include <memory>
#include <stdexcept>
#include <string>

#include <refem/linalg/SparseMatrix.hpp>
#include <refem/linalg/Vector.hpp>
#include <refem/solver/ReSolverLinearSolver.hpp>

#if defined(REFEM_HAS_RESOLVE)
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
    : preconditioner_("none"),
      A_(nullptr),
      workspace_type_(WorkspaceType::Cpu),
      workspace_(workspace)
  {
#if defined(REFEM_HAS_RESOLVE)
    resetSolver();
#endif
  }

  explicit Impl(ReSolve::LinAlgWorkspaceCUDA* workspace)
    : preconditioner_("none"),
      A_(nullptr),
      workspace_type_(WorkspaceType::Cuda),
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

    checkStatus(
        matrix_->copyFromExternal(A.rowPtrData(),
                                  A.colIndData(),
                                  A.valuesData(),
                                  ReSolve::memory::HOST,
                                  ReSolve::memory::HOST),
        "ReSolve Csr::copyFromExternal failed");

    checkStatus(solver_->setMatrix(matrix_.get()),
                "ReSolve SystemSolver::setMatrix failed");
    checkStatus(solver_->analyze(),
                "ReSolve SystemSolver::analyze failed");
    checkStatus(solver_->factorize(),
                "ReSolve SystemSolver::factorize failed");
#endif
  }

  void setPreconditioner(const std::string& method)
  {
    if (method.empty())
    {
      throw std::runtime_error("Preconditioner method must not be empty");
    }

    preconditioner_ = method;

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
    ReSolve::vector::Vector rhs(b.size());
    ReSolve::vector::Vector sol(x.size());

    checkStatus(rhs.copyFromExternal(b.data(),
                                     ReSolve::memory::HOST,
                                     ReSolve::memory::HOST),
                "ReSolve rhs Vector::copyFromExternal failed");
    checkStatus(sol.allocate(ReSolve::memory::HOST),
                "ReSolve solution Vector::allocate failed");

    checkStatus(solver_->solve(&rhs, &sol),
                "ReSolve SystemSolver::solve failed");
    checkStatus(sol.copyToExternal(x.data(),
                                   ReSolve::memory::HOST,
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
  void resetSolver()
  {
    switch (workspace_type_)
    {
    case WorkspaceType::Cpu:
      solver_ = std::make_unique<ReSolve::SystemSolver>(
          static_cast<ReSolve::LinAlgWorkspaceCpu*>(workspace_),
          "klu",
          "none",
          "klu",
          preconditioner_,
          "none");
      break;

    case WorkspaceType::Cuda:
#if defined(RESOLVE_USE_CUDA)
      solver_ = std::make_unique<ReSolve::SystemSolver>(
          static_cast<ReSolve::LinAlgWorkspaceCUDA*>(workspace_),
          "klu",
          "cusolverrf",
          "cusolverrf",
          preconditioner_,
          "none");
      break;
#else
      throw std::runtime_error(
          "This ReSolve installation was not built with CUDA support");
#endif
    }
  }
#endif

  enum class WorkspaceType
  {
    Cpu,
    Cuda,
  };

  std::string preconditioner_;

  const SparseMatrix* A_;
  WorkspaceType       workspace_type_;
  void*               workspace_;

#if defined(REFEM_HAS_RESOLVE)
  std::unique_ptr<ReSolve::SystemSolver>       solver_;
  std::unique_ptr<ReSolve::matrix::Csr>        matrix_;
#endif
};

ReSolveLinearSolver::ReSolveLinearSolver(ReSolve::LinAlgWorkspaceCpu* workspace)
  : impl_(std::make_unique<Impl>(workspace))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(ReSolve::LinAlgWorkspaceCUDA* workspace)
  : impl_(std::make_unique<Impl>(workspace))
{
}

ReSolveLinearSolver::ReSolveLinearSolver(ReSolve::LinAlgWorkspaceHIP* workspace)
  : impl_(std::make_unique<Impl>(workspace))
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
