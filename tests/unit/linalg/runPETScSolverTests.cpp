#include <petscksp.h>

#include "../state/StationaryFixtures.hpp"
#include "SolverTestFixtures.hpp"
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>
#include <femx/state/StateSolver.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx::linalg
{

void setScalar(PETScOperator& mat, Real val)
{
  if (mat.rows() != 1 || mat.cols() != 1)
  {
    mat.resize(1, 1);
  }
  mat.setZero();
  mat.set(0, 0, val);
}

} // namespace femx::linalg

namespace femx::tests
{
namespace
{

void configureKsp(linalg::KspLinearSolver& solver)
{
  auto& opts       = solver.opts();
  opts.type        = KSPGMRES;
  opts.pc_type     = PCNONE;
  opts.rtol        = 1.0e-12;
  opts.atol        = 1.0e-14;
  opts.max_its     = 30;
  opts.restart     = 10;
  opts.use_opts_db = false;
}

class ScalarTimeResidual final
  : public state::TimeResidual<linalg::PetscBackend>
{
public:
  explicit ScalarTimeResidual(Index num_steps)
    : dims_{num_steps, 1, 1, 1, 1},
      graph_(1,
             1,
             HostIndexVector{0, 1},
             HostIndexVector{0})
  {
  }

  state::TimeDims dims() const override
  {
    return dims_;
  }

  const HostCsrGraph& hostGraph() const override
  {
    return graph_;
  }

  const HostCsrGraph& graph() const override
  {
    return graph_;
  }

  void initialState(ConstView,
                    HostVector& out,
                    Ctx&) const override
  {
    out.assign(1, 0.0);
  }

  void res(const StepCtx& time,
           HostVector&    out,
           Ctx&) const override
  {
    out.assign(1, time.nxt[0] - time.hist.state(0)[0] - time.prm[0]);
  }

  void assembleNext(const StepCtx&         time,
                    HostVector&            out,
                    linalg::PETScOperator& jac,
                    Ctx&                   ctx) const override
  {
    res(time, out, ctx);
    if (jac.rows() != 1 || jac.cols() != 1)
    {
      jac.resize(1, 1);
    }
    jac.setZero();
    jac.set(0, 0, 1.0);
  }

  void applyJacT(const StepCtx&,
                 state::VariableBlock wrt,
                 ConstView            adj,
                 HostVector&          out,
                 Ctx&) const override
  {
    require(!wrt.isNextState(),
            "ScalarTimeResidual transpose apply supports history and parameters");
    out.assign(1, -adj[0]);
  }

private:
  state::TimeDims dims_;
  HostCsrGraph    graph_;
};

class FinalStateObjective final : public inverse::TimeObjective
{
public:
  explicit FinalStateObjective(Index num_steps, Real target)
    : num_steps_(num_steps), target_(target)
  {
  }

  Index numSteps() const override
  {
    return num_steps_;
  }

  Index numStates() const override
  {
    return 1;
  }

  Index numParams() const override
  {
    return 1;
  }

  Real value(const state::TimeTrajectory& tr,
             const HostVector&) const override
  {
    const Real diff = tr.level(num_steps_)[0] - target_;
    return 0.5 * diff * diff;
  }

  void stateGrad(Index                        level,
                 const state::TimeTrajectory& tr,
                 const HostVector&,
                 HostVector& out) const override
  {
    out.assign(1, level == num_steps_ ? tr.level(level)[0] - target_ : 0.0);
  }

  void paramGrad(const state::TimeTrajectory&,
                 const HostVector&,
                 HostVector& out) const override
  {
    out.assign(1, 0.0);
  }

private:
  Index num_steps_{0};
  Real  target_{0.0};
};

TestOutcome petscBackendSolvesForwardAndTranspose()
{
  TestStatus status(__func__);

  try
  {
    linalg::PETScOperator mat(PETSC_COMM_SELF);
    mat.resize(3, 3);
    solver::fillTestMat(mat);

    const HostVector expected = solver::expectedSolution();
    HostVector       rhs;
    HostVector       rhs_t;
    mat.apply(expected.view(), rhs);
    mat.applyT(expected.view(), rhs_t);

    linalg::KspLinearSolver lin_solver(PETSC_COMM_SELF);
    linalg::PetscContext    ctx{PETSC_COMM_SELF};
    configureKsp(lin_solver);

    HostVector actual;
    lin_solver.solve(mat, rhs, actual, ctx);
    status *= solver::vecNear(actual, expected, 1.0e-8);

    lin_solver.solveT(mat, rhs_t, actual, ctx);
    status *= solver::vecNear(actual, expected, 1.0e-8);
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }

  return status.report();
}

TestOutcome petscBackendRunsSharedTimeAdjoint()
{
  TestStatus status(__func__);

  try
  {
    constexpr Index         num_steps = 2;
    ScalarTimeResidual      res(num_steps);
    FinalStateObjective     obj(num_steps, 1.0);
    linalg::PETScOperator   fwd_jac(PETSC_COMM_SELF);
    linalg::PETScOperator   adj_jac(PETSC_COMM_SELF);
    linalg::KspLinearSolver fwd_solver(PETSC_COMM_SELF);
    linalg::KspLinearSolver adj_solver(PETSC_COMM_SELF);
    linalg::PetscContext    ctx{PETSC_COMM_SELF};
    configureKsp(fwd_solver);
    configureKsp(adj_solver);

    state::TimeIntegrator<linalg::PetscBackend> integ(
        res, fwd_jac, fwd_solver, ctx);
    inverse::TimeReducedFunctional<linalg::PetscBackend> reduced(
        integ, adj_jac, adj_solver, obj);

    const HostVector prm{0.25};
    HostVector       grad(1);
    const Real       val  = reduced.valueGrad(prm.view(), grad.view());
    status               *= std::abs(val - 0.125) < 1.0e-12;
    status               *= std::abs(grad[0] + 1.0) < 1.0e-10;

    constexpr Real   eps = 1.0e-6;
    const HostVector plus{prm[0] + eps};
    const HostVector minus{prm[0] - eps};
    const Real       fd = (reduced.value(plus.view())
                     - reduced.value(minus.view()))
                    / (2.0 * eps);
    status *= std::abs(grad[0] - fd) < 1.0e-8;
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }

  return status.report();
}

TestOutcome petscBackendRunsSharedStationaryAdjoint()
{
  TestStatus status(__func__);

  try
  {
    stationary::AffineResidual<linalg::PetscBackend> res(2.0);
    stationary::QuadraticObjective                   obj(1.0, 0.25);
    linalg::PETScOperator                            fwd_jac(PETSC_COMM_SELF);
    linalg::PETScOperator                            adj_jac(PETSC_COMM_SELF);
    linalg::KspLinearSolver                          fwd_solver(PETSC_COMM_SELF);
    linalg::KspLinearSolver                          adj_solver(PETSC_COMM_SELF);
    linalg::PetscContext                             ctx{PETSC_COMM_SELF};
    configureKsp(fwd_solver);
    configureKsp(adj_solver);

    state::LinearStateSolver<linalg::PetscBackend> state_solver(
        res, fwd_jac, fwd_solver, ctx);
    inverse::ReducedFunctional<linalg::PetscBackend> reduced(
        state_solver, adj_jac, adj_solver, obj);

    const HostVector prm{0.6};
    HostVector       grad;
    const Real       val  = reduced.valueGrad(prm, grad);
    status               *= std::abs(val - 0.29) < 1.0e-12;
    status               *= grad.size() == 1;
    status               *= std::abs(grad[0] + 0.2) < 1.0e-10;

    constexpr Real   eps = 1.0e-6;
    const HostVector plus{prm[0] + eps};
    const HostVector minus{prm[0] - eps};
    const Real       fd =
        (reduced.value(plus) - reduced.value(minus)) / (2.0 * eps);
    status *= std::abs(grad[0] - fd) < 1.0e-8;
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }

  return status.report();
}

} // namespace
} // namespace femx::tests

int main(int argc, char** argv)
{
  if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS)
  {
    return 1;
  }

  femx::tests::TestingResults results;
  results            += femx::tests::petscBackendSolvesForwardAndTranspose();
  results            += femx::tests::petscBackendRunsSharedTimeAdjoint();
  results            += femx::tests::petscBackendRunsSharedStationaryAdjoint();
  const int failures  = results.summary();

  PetscFinalize();
  return failures;
}
