#include <stdexcept>
#include <string>

#include <femx/common/Checks.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>

namespace femx
{
namespace linalg
{

using detail::check;
using detail::checkInit;
using detail::checkMPI;

class KspLinearSolver::Impl
{
private:
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

  void solve(const PETScOperator& op,
             const HostVector&    rhs,
             HostVector&          out)
  {
    solveSystem(op, rhs, out, false);
  }

  void solveT(const PETScOperator& op,
              const HostVector&    rhs,
              HostVector&          out)
  {
    solveSystem(op, rhs, out, true);
  }

  void checkContext(const PetscContext& ctx) const
  {
    checkSameComm(comm_, ctx.comm, "context");
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
  void solveSystem(const PETScOperator& op,
                   const HostVector&    rhs,
                   HostVector&          out,
                   bool                 tr)
  {
    require(op.rows() == op.cols(),
            "KspLinearSolver requires a square PETSc matrix");
    require(rhs.size() == op.rows(),
            "KspLinearSolver received inconsistent rhs size");

    checkSameComm(comm_, op.comm(), "matrix");

    const PetscInt size = static_cast<PetscInt>(op.rows());

    ScopedVec rhs_vec;
    ScopedVec out_vec;
    createVec(op.comm(), size, rhs_vec);
    check(VecDuplicate(rhs_vec.get(), out_vec.put()), "VecDuplicate");
    check(detail::copyToPETSc(rhs.view(), rhs_vec.get()), "copyToPETSc");
    setInitialGuess(out_vec.get(), out, op.rows());

    ensureKsp();
    configureKsp(ksp_);
    check(KSPSetOperators(ksp_, op.mat(), op.mat()), "KSPSetOperators");
    if (tr)
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

  static void createVec(MPI_Comm   comm,
                        PetscInt   size,
                        ScopedVec& vec)
  {
    PetscMPIInt comm_size = 1;
    checkMPI(MPI_Comm_size(comm, &comm_size), "MPI_Comm_size");
    const PetscInt num_local_dofs = comm_size == 1 ? size : PETSC_DECIDE;

    check(VecCreate(comm, vec.put()), "VecCreate");
    check(VecSetSizes(vec.get(), num_local_dofs, size), "VecSetSizes");
    check(VecSetFromOptions(vec.get()), "VecSetFromOptions");
  }

  void setInitialGuess(Vec               vec,
                       const HostVector& guess,
                       Index             size)
  {
    if (opts_.nonzero_guess && guess.size() == size)
    {
      check(detail::copyToPETSc(guess.view(), vec), "copyToPETSc");
      return;
    }
    check(VecSet(vec, 0.0), "VecSet");
  }

  static void checkSameComm(MPI_Comm    exp,
                            MPI_Comm    actual,
                            const char* object)
  {
    int result = MPI_UNEQUAL;
    if (MPI_Comm_compare(exp, actual, &result) != MPI_SUCCESS
        || (result != MPI_IDENT && result != MPI_CONGRUENT))
    {
      throw std::runtime_error(
          std::string("KspLinearSolver communicator mismatch for ") + object);
    }
  }

  void ensureKsp()
  {
    checkInit();
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

    MPI_Comm ksp_comm = MPI_COMM_NULL;
    check(PetscObjectGetComm(reinterpret_cast<PetscObject>(ksp), &ksp_comm),
          "PetscObjectGetComm");
    PetscMPIInt comm_size = 1;
    checkMPI(MPI_Comm_size(ksp_comm, &comm_size), "MPI_Comm_size");

    const std::string pc_type = opts_.pc_type == PCILU && comm_size > 1
                                    ? PCBJACOBI
                                    : opts_.pc_type;
    check(PCSetType(pc, pc_type.c_str()), "PCSetType");
    if (pc_type == PCILU)
    {
      check(PCFactorSetLevels(
                pc, static_cast<PetscInt>(opts_.factor_levels)),
            "PCFactorSetLevels");
    }
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
      throw std::runtime_error(
          "KspLinearSolver failed to converge: reason="
          + std::to_string(static_cast<int>(last_reason_))
          + ", iterations=" + std::to_string(static_cast<long long>(last_its_))
          + ", residual=" + std::to_string(last_rnorm_));
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
  : impl_(std::make_unique<Impl>(comm))
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

void KspLinearSolver::solve(const PETScOperator& mat,
                            const HostVector&    rhs,
                            HostVector&          sol,
                            PetscContext&        ctx)
{
  impl_->checkContext(ctx);
  impl_->solve(mat, rhs, sol);
}

void KspLinearSolver::solveT(const PETScOperator& mat,
                             const HostVector&    rhs,
                             HostVector&          sol,
                             PetscContext&        ctx)
{
  impl_->checkContext(ctx);
  impl_->solveT(mat, rhs, sol);
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
