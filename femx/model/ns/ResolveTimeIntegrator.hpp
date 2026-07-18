#pragma once

#include <memory>

#include <femx/common/Types.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx::inverse
{
class DeviceTimeObjective;
class TimeObjective;
class TimeReducedProgressMonitor;
} // namespace femx::inverse

namespace femx::model::ns
{

class NavierStokesModel;

/**
 * @brief Persistent CUDA time integrator for the ReSolve backend.
 *
 * Mesh data, maps, sparse storage, vectors, and ReSolve state are allocated
 * during construction and retained across solves. Host trajectories are
 * materialized only at the public output boundary.
 */
class ResolveTimeIntegrator final : public state::TimeIntegrator
{
public:
  /**
   * @brief Create a fixed-boundary CUDA integrator.
   *
   * Boundary data is copied once. Call setInitialState() before solving when
   * a nonzero initial state is required.
   */
  ResolveTimeIntegrator(const NavierStokesModel& model,
                        Array<Index>             bc_dofs,
                        HostVector               bc_vals,
                        linalg::ReSolveOptions   opts = {});

  /**
   * @brief Create a persistent controlled CUDA integrator.
   *
   * Control and initial-state topology is copied once during construction.
   * Parameters remain on Device throughout each solve.
   */
  ResolveTimeIntegrator(const NavierStokesModel& model,
                        fem::HostControlMap      ctr,
                        HostVector               init,
                        linalg::ReSolveOptions   opts = {});

  /**
   * @brief Create a controlled CUDA integrator with affine initial state.
   *
   * The initial map may depend on both modal and level-zero boundary-control
   * parameters. Its topology is copied once during construction.
   */
  ResolveTimeIntegrator(const NavierStokesModel& model,
                        fem::HostControlMap      ctr,
                        fem::HostInitialStateMap init,
                        linalg::ReSolveOptions   opts = {});

  ~ResolveTimeIntegrator() override;

  ResolveTimeIntegrator(const ResolveTimeIntegrator&)            = delete;
  ResolveTimeIntegrator& operator=(const ResolveTimeIntegrator&) = delete;

  /** @brief Number of residual time steps. */
  Index numSteps() const override;

  /** @brief Number of entries in one state. */
  Index numStates() const override;

  /** @brief Number of control and initial-state parameters. */
  Index numParams() const override;

  /** @brief Copy and retain the initial state on Device. */
  void setInitialState(const HostVector& state);

  /** @brief Solve on Device and copy the complete trajectory to Host. */
  void solve(const HostVector& prm, state::TimeTrajectory& tr) override;

  /** @brief Solve while retaining every state level on Device. */
  void solve(const HostVector& prm, state::DeviceTimeTrajectory& tr);

  /** @brief Accumulated Host-side assembly submission time. */
  Real assemblySeconds() const;

  /** @brief Accumulated ReSolve call time, including stream boundaries. */
  Real solveSeconds() const;

  /** @brief Number of residual/Jacobian assemblies. */
  Index assemblyCalls() const;

  /** @brief Number of forward or transpose linear solves. */
  Index solveCalls() const;

  /** @brief Reset accumulated timings and call counters. */
  void resetTiming();

private:
  friend class ResolveTimeReducedFunctional;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief CUDA ReSolve forward/adjoint functional for Navier--Stokes.
 *
 * The supplied integrator and Host objective must outlive this object. Built-in
 * objective metadata is flattened and copied once. Each evaluation transfers
 * only the input parameters, returned scalar, and requested gradient.
 * Evaluations mutate persistent work storage and are not reentrant.
 */
class ResolveTimeReducedFunctional final
{
public:
  /** @brief Bind one persistent integrator and built-in objective graph. */
  ResolveTimeReducedFunctional(ResolveTimeIntegrator&        integ,
                               const inverse::TimeObjective& obj);

  ~ResolveTimeReducedFunctional();

  ResolveTimeReducedFunctional(
      const ResolveTimeReducedFunctional&) = delete;
  ResolveTimeReducedFunctional& operator=(
      const ResolveTimeReducedFunctional&) = delete;

  /** @brief Number of reduced parameters. */
  Index numParams() const;

  /** @brief Run the Device forward solve and return the objective value. */
  Real value(const HostVector& prm);

  /** @brief Run forward and adjoint solves and copy the gradient to Host. */
  void grad(const HostVector& prm, HostVector& out);

  /** @brief Return the value and copy its reduced gradient to Host. */
  Real valueGrad(const HostVector& prm, HostVector& out);

  /** @brief Set a non-owning progress monitor. */
  void setMonitor(inverse::TimeReducedProgressMonitor* monitor);

  /** @brief Remove the current progress monitor. */
  void clearMonitor();

  /** @brief Forward accumulated assembly time and counters. */
  Real assemblySeconds() const;

  /** @brief Forward accumulated solve time. */
  Real solveSeconds() const;

  /** @brief Forward the number of assembly calls. */
  Index assemblyCalls() const;

  /** @brief Forward the number of linear solves. */
  Index solveCalls() const;

  /** @brief Reset the integrator timing counters. */
  void resetTiming();

private:
  void solveForward(const HostVector& prm);
  void solveAdjoint(HostVector& out);
  void notify(const char* phase, Index step);

  ResolveTimeIntegrator&                        integ_;
  std::unique_ptr<inverse::DeviceTimeObjective> obj_;
  inverse::TimeReducedProgressMonitor*          monitor_{nullptr};
};

} // namespace femx::model::ns
