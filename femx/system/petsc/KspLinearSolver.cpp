#include <stdexcept>
#include <string>

#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearOperator.hpp>
#include <femx/system/petsc/KspLinearSolver.hpp>
#include <femx/system/petsc/PETScSystemMatrix.hpp>
#include <femx/system/petsc/PETScSystemVector.hpp>
#include <femx/system/petsc/VectorConversion.hpp>

namespace femx
{
namespace system
{

class KspLinearSolver::Impl
{
private:
  struct ShellContext
  {
    const LinearOperator* op{nullptr};
    Vector<Real>          input;
    Vector<Real>          output;
  };

  class ScopedVec
  {
  public:
    ~ScopedVec()
    {
      if (vec_ != nullptr)
      {
        VecDestroy(&vec_);
      }
    }

    Vec get() const
    {
      return vec_;
    }

    Vec* put()
    {
      return &vec_;
    }

  private:
    Vec vec_{nullptr};
  };

  class ScopedMat
  {
  public:
    ~ScopedMat()
    {
      if (mat_ != nullptr)
      {
        MatDestroy(&mat_);
      }
    }

    Mat get() const
    {
      return mat_;
    }

    Mat* put()
    {
      return &mat_;
    }

  private:
    Mat mat_{nullptr};
  };

  class ScopedKsp
  {
  public:
    ~ScopedKsp()
    {
      if (ksp_ != nullptr)
      {
        KSPDestroy(&ksp_);
      }
    }

    KSP get() const
    {
      return ksp_;
    }

    KSP* put()
    {
      return &ksp_;
    }

  private:
    KSP ksp_{nullptr};
  };

public:
  explicit Impl(MPI_Comm comm)
    : comm_(comm)
  {
  }

  ~Impl()
  {
    if (ksp_ != nullptr)
    {
      KSPDestroy(&ksp_);
    }
  }

  KspOptions& options()
  {
    return options_;
  }

  const KspOptions& options() const
  {
    return options_;
  }

  void solve(const LinearOperator& op,
             const Vector<Real>&   rhs,
             Vector<Real>&         out)
  {
    solveLinearOperator(op, rhs, out, false);
  }

  void solveT(const LinearOperator& op,
              const Vector<Real>&   rhs,
              Vector<Real>&         out)
  {
    solveLinearOperator(op, rhs, out, true);
  }

  void solve(const PETScSystemMatrix& op,
             const PETScSystemVector& rhs,
             PETScSystemVector&       out)
  {
    solveSystem(op, rhs, out, false);
  }

  void solveT(const PETScSystemMatrix& op,
              const PETScSystemVector& rhs,
              PETScSystemVector&       out)
  {
    solveSystem(op, rhs, out, true);
  }

  KSPConvergedReason convergedReason() const
  {
    return last_reason_;
  }

  PetscInt its() const
  {
    return last_its_;
  }

  PetscReal rnorm() const
  {
    return last_rnorm_;
  }

private:
  void solveLinearOperator(const LinearOperator& op,
                           const Vector<Real>&   rhs,
                           Vector<Real>&         out,
                           bool                  transpose)
  {
    if (const auto* petsc_mat =
            dynamic_cast<const PETScSystemMatrix*>(&op))
    {
      solveSystem(*petsc_mat, rhs, out, transpose);
      return;
    }

    if (op.numRows() != op.numCols())
    {
      throw std::runtime_error("KspLinearSolver requires a square operator");
    }
    if (rhs.size() != op.numRows())
    {
      throw std::runtime_error(
          "KspLinearSolver received inconsistent rhs size");
    }

    checkInitialized();

    const PetscInt size = static_cast<PetscInt>(op.numRows());

    ShellContext context;
    context.op = &op;

    ScopedVec rhs_vec;
    ScopedVec out_vec;
    ScopedMat shell;
    ScopedKsp ksp;

    check(VecCreateSeq(PETSC_COMM_SELF, size, rhs_vec.put()), "VecCreateSeq");
    check(VecDuplicate(rhs_vec.get(), out_vec.put()), "VecDuplicate");
    check(detail::copyToPETSc(rhs, rhs_vec.get()), "copyToPETSc");
    setInitialGuess(out_vec.get(), out, op.numRows());

    Mat mat = matrixForOperator(op, context, shell, size);

    check(KSPCreate(PETSC_COMM_SELF, ksp.put()), "KSPCreate");
    configureKsp(ksp.get());
    check(KSPSetOperators(ksp.get(), mat, mat), "KSPSetOperators");

    if (transpose)
    {
      check(KSPSolveTranspose(ksp.get(), rhs_vec.get(), out_vec.get()),
            "KSPSolveTranspose");
    }
    else
    {
      check(KSPSolve(ksp.get(), rhs_vec.get(), out_vec.get()), "KSPSolve");
    }

    updateStats(ksp.get());
    checkConverged();

    check(detail::copyFromPETSc(out_vec.get(), out), "copyFromPETSc");
  }

  void solveSystem(const PETScSystemMatrix& op,
                   const Vector<Real>&      rhs,
                   Vector<Real>&            out,
                   bool                     transpose)
  {
    if (op.numRows() != op.numCols())
    {
      throw std::runtime_error(
          "KspLinearSolver requires a square PETSc matrix");
    }
    if (rhs.size() != op.numRows())
    {
      throw std::runtime_error(
          "KspLinearSolver received inconsistent rhs size");
    }

    checkSameComm(comm_, op.comm(), "matrix");

    const PetscInt size = static_cast<PetscInt>(op.numRows());

    ScopedVec rhs_vec;
    ScopedVec out_vec;
    createVec(op.comm(), size, rhs_vec);
    check(VecDuplicate(rhs_vec.get(), out_vec.put()), "VecDuplicate");
    check(detail::copyToPETSc(rhs, rhs_vec.get()), "copyToPETSc");
    setInitialGuess(out_vec.get(), out, op.numRows());

    ensureKsp();
    configureKsp(ksp_);
    check(KSPSetOperators(ksp_, op.mat(), op.mat()), "KSPSetOperators");

    if (transpose)
    {
      check(KSPSolveTranspose(ksp_, rhs_vec.get(), out_vec.get()),
            "KSPSolveTranspose");
    }
    else
    {
      check(KSPSolve(ksp_, rhs_vec.get(), out_vec.get()), "KSPSolve");
    }

    updateStats(ksp_);
    checkConverged();

    check(detail::copyFromPETSc(out_vec.get(), out), "copyFromPETSc");
  }

  void solveSystem(const PETScSystemMatrix& op,
                   const PETScSystemVector& rhs,
                   PETScSystemVector&       out,
                   bool                     transpose)
  {
    if (op.numRows() != op.numCols())
    {
      throw std::runtime_error(
          "KspLinearSolver requires a square PETSc matrix");
    }

    const Index rhs_size = transpose ? op.numCols() : op.numRows();
    const Index out_size = transpose ? op.numRows() : op.numCols();
    if (rhs.size() != rhs_size || out.size() != out_size)
    {
      throw std::runtime_error(
          "KspLinearSolver received incompatible PETSc vectors");
    }

    checkSameComm(comm_, op.comm(), "matrix");
    checkSameComm(comm_, rhs.comm(), "right-hand side");
    checkSameComm(comm_, out.comm(), "solution");

    ensureKsp();
    configureKsp(ksp_);
    check(KSPSetOperators(ksp_, op.mat(), op.mat()), "KSPSetOperators");

    if (transpose)
    {
      check(KSPSolveTranspose(ksp_, rhs.vec(), out.vec()),
            "KSPSolveTranspose");
    }
    else
    {
      check(KSPSolve(ksp_, rhs.vec(), out.vec()), "KSPSolve");
    }

    updateStats(ksp_);
    checkConverged();
  }

  static Mat matrixForOperator(const LinearOperator& op,
                               ShellContext&         context,
                               ScopedMat&            shell,
                               PetscInt              size)
  {
    if (const auto* petsc_mat =
            dynamic_cast<const PETScSystemMatrix*>(&op))
    {
      return petsc_mat->mat();
    }

    check(MatCreateShell(PETSC_COMM_SELF,
                         size,
                         size,
                         size,
                         size,
                         &context,
                         shell.put()),
          "MatCreateShell");
    check(MatShellSetOperation(shell.get(),
                               MATOP_MULT,
                               reinterpret_cast<void (*)(void)>(matMult)),
          "MatShellSetOperation(MATOP_MULT)");
    check(MatShellSetOperation(
              shell.get(),
              MATOP_MULT_TRANSPOSE,
              reinterpret_cast<void (*)(void)>(matMultTranspose)),
          "MatShellSetOperation(MATOP_MULT_TRANSPOSE)");
    return shell.get();
  }

  static void createVec(MPI_Comm   comm,
                        PetscInt   size,
                        ScopedVec& vec)
  {
    PetscMPIInt comm_size = 1;
    checkMPI(MPI_Comm_size(comm, &comm_size), "MPI_Comm_size");
    const PetscInt local_size = comm_size == 1 ? size : PETSC_DECIDE;

    check(VecCreate(comm, vec.put()), "VecCreate");
    check(VecSetSizes(vec.get(), local_size, size), "VecSetSizes");
    check(VecSetFromOptions(vec.get()), "VecSetFromOptions");
  }

  void setInitialGuess(Vec                 vec,
                       const Vector<Real>& guess,
                       Index               size)
  {
    if (options_.nonzero_guess && guess.size() == size)
    {
      check(detail::copyToPETSc(guess, vec), "copyToPETSc");
      return;
    }
    check(VecSet(vec, 0.0), "VecSet");
  }

  static PetscErrorCode matMult(Mat shell, Vec x, Vec y)
  {
    ShellContext* context = nullptr;
    PetscCall(MatShellGetContext(shell, &context));
    if (context == nullptr || context->op == nullptr)
    {
      return PETSC_ERR_ARG_NULL;
    }

    try
    {
      PetscCall(detail::copyFromPETSc(x, context->input));
      context->op->apply(context->input, context->output);
      PetscCall(detail::copyToPETSc(context->output, y));
    }
    catch (...)
    {
      return PETSC_ERR_LIB;
    }

    return PETSC_SUCCESS;
  }

  static PetscErrorCode matMultTranspose(Mat shell, Vec x, Vec y)
  {
    ShellContext* context = nullptr;
    PetscCall(MatShellGetContext(shell, &context));
    if (context == nullptr || context->op == nullptr)
    {
      return PETSC_ERR_ARG_NULL;
    }

    try
    {
      PetscCall(detail::copyFromPETSc(x, context->input));
      context->op->applyT(context->input, context->output);
      PetscCall(detail::copyToPETSc(context->output, y));
    }
    catch (...)
    {
      return PETSC_ERR_LIB;
    }

    return PETSC_SUCCESS;
  }

  static void check(PetscErrorCode ierr, const char* operation)
  {
    if (ierr != PETSC_SUCCESS)
    {
      throw std::runtime_error(std::string(operation) + " failed");
    }
  }

  static void checkMPI(int ierr, const char* operation)
  {
    if (ierr != MPI_SUCCESS)
    {
      throw std::runtime_error(std::string(operation) + " failed");
    }
  }

  static void checkInitialized()
  {
    PetscBool initialized = PETSC_FALSE;
    check(PetscInitialized(&initialized), "PetscInitialized");
    if (initialized != PETSC_TRUE)
    {
      throw std::runtime_error("KspLinearSolver requires initialized PETSc");
    }
  }

  static void checkSameComm(MPI_Comm    expected,
                            MPI_Comm    actual,
                            const char* object)
  {
    int result = MPI_UNEQUAL;
    if (MPI_Comm_compare(expected, actual, &result) != MPI_SUCCESS
        || (result != MPI_IDENT && result != MPI_CONGRUENT))
    {
      throw std::runtime_error(
          std::string("KspLinearSolver communicator mismatch for ") + object);
    }
  }

  void ensureKsp()
  {
    checkInitialized();
    if (ksp_ == nullptr)
    {
      check(KSPCreate(comm_, &ksp_), "KSPCreate");
    }
  }

  void configureKsp(KSP ksp) const
  {
    check(KSPSetType(ksp, options_.type.c_str()), "KSPSetType");
    if (options_.restart > 0
        && (options_.type == KSPGMRES || options_.type == KSPFGMRES))
    {
      check(KSPGMRESSetRestart(ksp,
                               static_cast<PetscInt>(options_.restart)),
            "KSPGMRESSetRestart");
    }

    PC pc = nullptr;
    check(KSPGetPC(ksp, &pc), "KSPGetPC");
    check(PCSetType(pc, options_.pc_type.c_str()), "PCSetType");
    check(KSPSetTolerances(
              ksp,
              static_cast<PetscReal>(options_.rtol),
              static_cast<PetscReal>(options_.atol),
              static_cast<PetscReal>(options_.dtol),
              static_cast<PetscInt>(options_.max_its)),
          "KSPSetTolerances");
    check(KSPSetInitialGuessNonzero(
              ksp,
              options_.nonzero_guess ? PETSC_TRUE : PETSC_FALSE),
          "KSPSetInitialGuessNonzero");
    if (options_.use_opts_db)
    {
      check(KSPSetFromOptions(ksp), "KSPSetFromOptions");
    }
  }

  void updateStats(KSP ksp)
  {
    check(KSPGetConvergedReason(ksp, &last_reason_),
          "KSPGetConvergedReason");
    check(KSPGetIterationNumber(ksp, &last_its_),
          "KSPGetIterationNumber");
    check(KSPGetResidualNorm(ksp, &last_rnorm_),
          "KSPGetResidualNorm");
  }

  void checkConverged() const
  {
    if (static_cast<int>(last_reason_) <= 0)
    {
      throw std::runtime_error("KspLinearSolver failed to converge");
    }
  }

private:
  MPI_Comm           comm_{PETSC_COMM_SELF};
  KSP                ksp_{nullptr};
  KspOptions         options_;
  KSPConvergedReason last_reason_{KSP_CONVERGED_ITERATING};
  PetscInt           last_its_{0};
  PetscReal          last_rnorm_{0.0};
};

KspLinearSolver::KspLinearSolver(MPI_Comm comm)
  : impl_(std::make_unique<Impl>(comm))
{
}

KspLinearSolver::~KspLinearSolver() = default;

KspOptions& KspLinearSolver::options()
{
  return impl_->options();
}

const KspOptions& KspLinearSolver::options() const
{
  return impl_->options();
}

void KspLinearSolver::solve(const LinearOperator& op,
                            const Vector<Real>&   rhs,
                            Vector<Real>&         out)
{
  impl_->solve(op, rhs, out);
}

void KspLinearSolver::solveT(const LinearOperator& op,
                             const Vector<Real>&   rhs,
                             Vector<Real>&         out)
{
  impl_->solveT(op, rhs, out);
}

void KspLinearSolver::solve(const PETScSystemMatrix& op,
                            const PETScSystemVector& rhs,
                            PETScSystemVector&       out)
{
  impl_->solve(op, rhs, out);
}

void KspLinearSolver::solveT(const PETScSystemMatrix& op,
                             const PETScSystemVector& rhs,
                             PETScSystemVector&       out)
{
  impl_->solveT(op, rhs, out);
}

KSPConvergedReason KspLinearSolver::convergedReason() const
{
  return impl_->convergedReason();
}

PetscInt KspLinearSolver::its() const
{
  return impl_->its();
}

PetscReal KspLinearSolver::rnorm() const
{
  return impl_->rnorm();
}

} // namespace system
} // namespace femx
