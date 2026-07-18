#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/CsrTranspose.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>
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

namespace femx
{
namespace linalg
{

class ReSolveLinearSolver::Impl
{
public:
  Impl()
  {
#if defined(FEMX_HAS_RESOLVE)
    initWork();
    resetSolver();
#endif
  }

  explicit Impl(ReSolveOptions opts)
    : opts_(std::move(opts))
  {
#if defined(FEMX_HAS_RESOLVE)
    initWork();
    resetSolver();
#endif
  }

  void setOperator(const HostCsrMatrix& A)
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
      checkStatus(mat_->allocateAll(ReSolve::memory::HOST),
                  "ReSolve Csr::allocateAll failed");
      mat_rows_ = A.rows();
      mat_cols_ = A.cols();
      mat_nnz_  = A.nnz();
    }

    updateMat(A, *mat_, "ReSolve");

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
      setup(*solver_, "ReSolve");
    }
#endif
  }

  void solve(const HostVector& b, HostVector& x)
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
    if (isZero(b))
    {
      return;
    }

#if defined(FEMX_HAS_RESOLVE)
    solveWith(*solver_, b, x, "ReSolve SystemSolver::solve failed");
#else
    throw std::runtime_error(
        "ReSolveLinearSolver was built without ReSolve support");
#endif
  }

  void solve(const LinearOperator& op, const HostVector& rhs, HostVector& out)
  {
    const MapCsrMatrix& csr = requireMapCsrMatrix(op);
    if (op.numRows() != op.numCols() || rhs.size() != op.numRows())
    {
      throw std::runtime_error(
          "ReSolveLinearSolver received inconsistent operator solve dimensions");
    }

    setOperator(csr.mat());
    solve(rhs, out);
  }

  void solveT(const LinearOperator& op, const HostVector& rhs, HostVector& out)
  {
    const MapCsrMatrix& csr = requireMapCsrMatrix(op);
    if (op.numRows() != op.numCols() || rhs.size() != op.numCols())
    {
      throw std::runtime_error(
          "ReSolveLinearSolver received inconsistent transpose solve dimensions");
    }

#if defined(FEMX_HAS_RESOLVE)
    solveTr(csr.mat(), rhs, out);
#else
    throw std::runtime_error(
        "ReSolveLinearSolver was built without ReSolve support");
#endif
  }

private:
  static const MapCsrMatrix& requireMapCsrMatrix(
      const LinearOperator& op)
  {
    const auto* csr = dynamic_cast<const MapCsrMatrix*>(&op);
    if (csr == nullptr)
    {
      throw std::runtime_error(
          "ReSolveLinearSolver currently supports MapCsrMatrix only");
    }
    return *csr;
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
  void solveTr(const HostCsrMatrix& A,
               const HostVector&    b,
               HostVector&          x)
  {
    if (A.rows() != A.cols() || b.size() != A.cols())
    {
      throw std::runtime_error(
          "ReSolveLinearSolver received inconsistent transpose dimensions");
    }
    resizeOrZero(x, A.rows());
    if (isZero(b))
    {
      return;
    }

    setTrOperator(A);
    solveWith(*tr_solver_,
              b,
              x,
              "ReSolve transpose SystemSolver::solve failed");
  }

  void setTrOperator(const HostCsrMatrix& A)
  {
    if (tr_map_ == nullptr
        || tr_map_->srcGraph().layoutId() != A.graph().layoutId())
    {
      tr_map_  = std::make_unique<HostCsrTransposeMap>(A.graph());
      tr_data_ = std::make_unique<HostCsrMatrix>(tr_map_->trGraph());
    }
    trVals(A, *tr_map_, *tr_data_);

    const bool reuse_mat =
        tr_mat_ != nullptr
        && tr_rows_ == A.cols()
        && tr_cols_ == A.rows()
        && tr_nnz_ == A.nnz()
        && opts_.factor == "none"
        && opts_.refactor == "none";

    if (!reuse_mat)
    {
      resetTrSolver();

      tr_mat_ = std::make_unique<ReSolve::matrix::Csr>(
          A.cols(),
          A.rows(),
          A.nnz());
      checkStatus(tr_mat_->allocateAll(ReSolve::memory::HOST),
                  "ReSolve transpose Csr::allocateAll failed");
      tr_rows_ = A.cols();
      tr_cols_ = A.rows();
      tr_nnz_  = A.nnz();
    }

    updateMat(*tr_data_, *tr_mat_, "ReSolve transpose");

    if (reuse_mat)
    {
      if (opts_.precond != "none")
      {
        checkStatus(
            tr_solver_->resetPreconditioner(tr_mat_.get()),
            "ReSolve transpose SystemSolver::resetPreconditioner failed");
      }
    }
    else
    {
      checkStatus(tr_solver_->setMatrix(tr_mat_.get()),
                  "ReSolve transpose SystemSolver::setMatrix failed");
      setup(*tr_solver_, "ReSolve transpose");
    }
  }

  static bool isZero(const HostVector& vals)
  {
    return std::all_of(vals.begin(),
                       vals.end(),
                       [](Real val)
                       {
                         return val == 0.0;
                       });
  }

  void updateMat(const HostCsrMatrix&  A,
                 ReSolve::matrix::Csr& mat,
                 const char*           prefix)
  {
    checkStatus(mat.copyFromExternal(A.rowPtrData(),
                                     A.colIndData(),
                                     A.valsData(),
                                     ReSolve::memory::HOST,
                                     ReSolve::memory::HOST),
                std::string(prefix) + " Csr::copyFromExternal failed");
  }

  void resetSolver()
  {
    solver_ = makeSolver();
    applyOpts(*solver_);
  }

  void resetTrSolver()
  {
    tr_solver_ = makeSolver();
    applyOpts(*tr_solver_);
  }

  std::unique_ptr<ReSolve::SystemSolver> makeSolver()
  {
    return std::make_unique<ReSolve::SystemSolver>(
        cpu_work_.get(),
        opts_.factor,
        opts_.refactor,
        opts_.solve,
        opts_.precond,
        opts_.ir);
  }

  void initWork()
  {
    cpu_work_ = std::make_unique<ReSolve::LinAlgWorkspaceCpu>();
    cpu_work_->initializeHandles();
  }

  void setup(ReSolve::SystemSolver& solver, const char* prefix)
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
                 const HostVector&      b,
                 HostVector&            x,
                 const char*            op)
  {
    constexpr auto memspace = ReSolve::memory::HOST;

    ReSolve::vector::Vector rhs(b.size());
    ReSolve::vector::Vector sol(x.size());

    checkStatus(rhs.allocate(memspace),
                "ReSolve rhs HostVector::allocate failed");
    checkStatus(rhs.copyFromExternal(b.data(), ReSolve::memory::HOST, memspace),
                "ReSolve rhs HostVector::copyFromExternal failed");

    checkStatus(sol.allocate(memspace),
                "ReSolve solution HostVector::allocate failed");
    checkStatus(sol.setToZero(memspace),
                "ReSolve solution HostVector::setToZero failed");

    checkStatus(solver.solve(&rhs, &sol), op);

    checkStatus(sol.copyToExternal(x.data(),
                                   memspace,
                                   ReSolve::memory::HOST),
                "ReSolve solution HostVector::copyToExternal failed");
  }

  void applyOpts(ReSolve::SystemSolver& solver)
  {
    if (opts_.solve == "fgmres" || opts_.solve == "randgmres")
    {
      solver.setGramSchmidtMethod(opts_.gram_schmidt);
      solver.getIterativeSolver().setMaxit(opts_.max_its);
      solver.getIterativeSolver().setTol(opts_.rtol);
      solver.getIterativeSolver().setCliParam(
          "restart",
          std::to_string(opts_.restart));
      solver.getIterativeSolver().setCliParam(
          "flexible",
          opts_.flexible ? "yes" : "no");

      if (opts_.solve == "randgmres")
      {
        checkStatus(solver.setSketchingMethod(opts_.sketching),
                    "ReSolve SystemSolver::setSketchingMethod failed");
      }
    }
  }
#endif

  ReSolveOptions                       opts_;
  const HostCsrMatrix*                 A_{nullptr};
  Index                                mat_rows_ = 0;
  Index                                mat_cols_ = 0;
  Index                                mat_nnz_  = 0;
  std::unique_ptr<HostCsrTransposeMap> tr_map_;
  std::unique_ptr<HostCsrMatrix>       tr_data_;
  Index                                tr_rows_ = 0;
  Index                                tr_cols_ = 0;
  Index                                tr_nnz_  = 0;

#if defined(FEMX_HAS_RESOLVE)
  std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> cpu_work_;
  std::unique_ptr<ReSolve::SystemSolver>       solver_;
  std::unique_ptr<ReSolve::matrix::Csr>        mat_;
  std::unique_ptr<ReSolve::SystemSolver>       tr_solver_;
  std::unique_ptr<ReSolve::matrix::Csr>        tr_mat_;
#endif
};

ReSolveLinearSolver::ReSolveLinearSolver()
  : impl_(std::make_unique<Impl>())
{
}

ReSolveLinearSolver::ReSolveLinearSolver(ReSolveOptions opts)
  : impl_(std::make_unique<Impl>(std::move(opts)))
{
}

ReSolveLinearSolver::~ReSolveLinearSolver() = default;

void ReSolveLinearSolver::setOperator(const HostCsrMatrix& A)
{
  impl_->setOperator(A);
}

void ReSolveLinearSolver::solve(const LinearOperator& op,
                                const HostVector&     rhs,
                                HostVector&           out)
{
  impl_->solve(op, rhs, out);
}

void ReSolveLinearSolver::solveT(const LinearOperator& op,
                                 const HostVector&     rhs,
                                 HostVector&           out)
{
  impl_->solveT(op, rhs, out);
}

void ReSolveLinearSolver::solve(const HostVector& b, HostVector& x)
{
  impl_->solve(b, x);
}

} // namespace linalg
} // namespace femx
