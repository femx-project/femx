#include <stdexcept>
#include <string>

#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScMatrixOperator.hpp>
#include <femx/linalg/petsc/PETScVectorBuilder.hpp>
#include <femx/linalg/petsc/VectorConversion.hpp>

using namespace std;

namespace femx
{
namespace linalg
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

  KspOptions& opts()
  {
    return opts_;
  }

  const KspOptions& opts() const
  {
    return opts_;
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

  void solve(const PETScMatrixOperator& op,
             const PETScVectorBuilder&  rhs,
             PETScVectorBuilder&        out)
  {
    solveSystem(op, rhs, out, false);
  }

  void solveT(const PETScMatrixOperator& op,
              const PETScVectorBuilder&  rhs,
              PETScVectorBuilder&        out)
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
            dynamic_cast<const PETScMatrixOperator*>(&op))
    {
      solveSystem(*petsc_mat, rhs, out, transpose);
      return;
    }

    if (op.numRows() != op.numCols())
    {
      throw runtime_error("KspLinearSolver requires a square operator");
    }
    if (rhs.size() != op.numRows())
    {
      throw runtime_error(
          "KspLinearSolver received inconsistent rhs size");
    }

    checkInitialized();

    const PetscInt size = static_cast<PetscInt>(op.numRows());

    ShellContext ctx;
    ctx.op = &op;

    ScopedVec rhs_vec;
    ScopedVec out_vec;
    ScopedMat shell;
    ScopedKsp ksp;

    check(VecCreateSeq(PETSC_COMM_SELF, size, rhs_vec.put()), "VecCreateSeq");
    check(VecDuplicate(rhs_vec.get(), out_vec.put()), "VecDuplicate");
    check(detail::copyToPETSc(rhs, rhs_vec.get()), "copyToPETSc");
    setInitialGuess(out_vec.get(), out, op.numRows());

    Mat mat = matrixForOperator(op, ctx, shell, size);

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

  void solveSystem(const PETScMatrixOperator& op,
                   const Vector<Real>&        rhs,
                   Vector<Real>&              out,
                   bool                       transpose)
  {
    if (op.numRows() != op.numCols())
    {
      throw runtime_error(
          "KspLinearSolver requires a square PETSc matrix");
    }
    if (rhs.size() != op.numRows())
    {
      throw runtime_error(
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

  void solveSystem(const PETScMatrixOperator& op,
                   const PETScVectorBuilder&  rhs,
                   PETScVectorBuilder&        out,
                   bool                       transpose)
  {
    if (op.numRows() != op.numCols())
    {
      throw runtime_error(
          "KspLinearSolver requires a square PETSc matrix");
    }

    const Index rhs_size = transpose ? op.numCols() : op.numRows();
    const Index out_size = transpose ? op.numRows() : op.numCols();
    if (rhs.size() != rhs_size || out.size() != out_size)
    {
      throw runtime_error(
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
                               ShellContext&         ctx,
                               ScopedMat&            shell,
                               PetscInt              size)
  {
    if (const auto* petsc_mat =
            dynamic_cast<const PETScMatrixOperator*>(&op))
    {
      return petsc_mat->mat();
    }

    check(MatCreateShell(PETSC_COMM_SELF,
                         size,
                         size,
                         size,
                         size,
                         &ctx,
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
    const PetscInt nloc = comm_size == 1 ? size : PETSC_DECIDE;

    check(VecCreate(comm, vec.put()), "VecCreate");
    check(VecSetSizes(vec.get(), nloc, size), "VecSetSizes");
    check(VecSetFromOptions(vec.get()), "VecSetFromOptions");
  }

  void setInitialGuess(Vec                 vec,
                       const Vector<Real>& guess,
                       Index               size)
  {
    if (opts_.nonzero_guess && guess.size() == size)
    {
      check(detail::copyToPETSc(guess, vec), "copyToPETSc");
      return;
    }
    check(VecSet(vec, 0.0), "VecSet");
  }

  static PetscErrorCode matMult(Mat shell, Vec x, Vec y)
  {
    ShellContext* ctx = nullptr;
    PetscCall(MatShellGetContext(shell, &ctx));
    if (ctx == nullptr || ctx->op == nullptr)
    {
      return PETSC_ERR_ARG_NULL;
    }

    try
    {
      PetscCall(detail::copyFromPETSc(x, ctx->input));
      ctx->op->apply(ctx->input, ctx->output);
      PetscCall(detail::copyToPETSc(ctx->output, y));
    }
    catch (...)
    {
      return PETSC_ERR_LIB;
    }

    return PETSC_SUCCESS;
  }

  static PetscErrorCode matMultTranspose(Mat shell, Vec x, Vec y)
  {
    ShellContext* ctx = nullptr;
    PetscCall(MatShellGetContext(shell, &ctx));
    if (ctx == nullptr || ctx->op == nullptr)
    {
      return PETSC_ERR_ARG_NULL;
    }

    try
    {
      PetscCall(detail::copyFromPETSc(x, ctx->input));
      ctx->op->applyT(ctx->input, ctx->output);
      PetscCall(detail::copyToPETSc(ctx->output, y));
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
      throw runtime_error(string(operation) + " failed");
    }
  }

  static void checkMPI(int ierr, const char* operation)
  {
    if (ierr != MPI_SUCCESS)
    {
      throw runtime_error(string(operation) + " failed");
    }
  }

  static void checkInitialized()
  {
    PetscBool initialized = PETSC_FALSE;
    check(PetscInitialized(&initialized), "PetscInitialized");
    if (initialized != PETSC_TRUE)
    {
      throw runtime_error("KspLinearSolver requires initialized PETSc");
    }
  }

  static void checkSameComm(MPI_Comm    exp,
                            MPI_Comm    actual,
                            const char* object)
  {
    int result = MPI_UNEQUAL;
    if (MPI_Comm_compare(exp, actual, &result) != MPI_SUCCESS
        || (result != MPI_IDENT && result != MPI_CONGRUENT))
    {
      throw runtime_error(
          string("KspLinearSolver communicator mismatch for ") + object);
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
    check(KSPSetType(ksp, opts_.type.c_str()), "KSPSetType");
    if (opts_.restart > 0
        && (opts_.type == KSPGMRES || opts_.type == KSPFGMRES))
    {
      check(KSPGMRESSetRestart(ksp,
                               static_cast<PetscInt>(opts_.restart)),
            "KSPGMRESSetRestart");
    }

    PC pc = nullptr;
    check(KSPGetPC(ksp, &pc), "KSPGetPC");
    check(PCSetType(pc, opts_.pc_type.c_str()), "PCSetType");
    check(KSPSetTolerances(
              ksp,
              static_cast<PetscReal>(opts_.rtol),
              static_cast<PetscReal>(opts_.atol),
              static_cast<PetscReal>(opts_.dtol),
              static_cast<PetscInt>(opts_.max_its)),
          "KSPSetTolerances");
    check(KSPSetInitialGuessNonzero(
              ksp,
              opts_.nonzero_guess ? PETSC_TRUE : PETSC_FALSE),
          "KSPSetInitialGuessNonzero");
    if (opts_.use_opts_db)
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
      throw runtime_error("KspLinearSolver failed to converge");
    }
  }

private:
  MPI_Comm           comm_{PETSC_COMM_SELF};
  KSP                ksp_{nullptr};
  KspOptions         opts_;
  KSPConvergedReason last_reason_{KSP_CONVERGED_ITERATING};
  PetscInt           last_its_{0};
  PetscReal          last_rnorm_{0.0};
};

KspLinearSolver::KspLinearSolver(MPI_Comm comm)
  : impl_(make_unique<Impl>(comm))
{
}

KspLinearSolver::~KspLinearSolver() = default;

KspOptions& KspLinearSolver::opts()
{
  return impl_->opts();
}

const KspOptions& KspLinearSolver::opts() const
{
  return impl_->opts();
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

void KspLinearSolver::solve(const PETScMatrixOperator& op,
                            const PETScVectorBuilder&  rhs,
                            PETScVectorBuilder&        out)
{
  impl_->solve(op, rhs, out);
}

void KspLinearSolver::solveT(const PETScMatrixOperator& op,
                             const PETScVectorBuilder&  rhs,
                             PETScVectorBuilder&        out)
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

} // namespace linalg
} // namespace femx
