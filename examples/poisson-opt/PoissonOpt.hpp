#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include <femx/assembly/ElementKernel.hpp>
#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/common/Workspace.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/inverse/LeastSquaresObjective.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/inverse/SumObjective.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/Residual.hpp>

namespace femx::state
{
class Linearization;
}

namespace femx::linalg
{
class LinearSolver;
}

namespace femx::state
{
class StateSolver;
}

namespace femx::examples::poisson_opt
{

struct Options
{
  Index         num_x_cells  = 32;
  Index         num_y_cells  = 32;
  WorkspaceType backend      = WorkspaceType::Cpu;
  bool          write_output = false;
  Real          alpha        = 1.0e-6;
  Index         obs_stride   = 0;
  Index         max_its      = 50;
};

/**
 * @brief Final metrics reported by the Poisson optimization example.
 *
 * Report combines objective, gradient, state-error, and control-error metrics
 * for the optimized solution.
 */
struct Report
{
  Real value             = 0.0;
  Real grad_norm         = 0.0;
  Real state_rms_error   = 0.0;
  Real state_max_err     = 0.0;
  Real ctr_rms_error     = 0.0;
  Real control_max_error = 0.0;
};

/**
 * @brief Optimizer result together with final control and state vectors.
 *
 * Result carries the final optimization state plus the TAO iteration count and
 * convergence status returned by the driver.
 */
struct Result
{
  Report       report;
  Vector<Real> prm;
  Vector<Real> state;
  Index        tao_itr    = 0;
  int          tao_reason = 0;
  bool         converged  = false;
};

/**
 * @brief Poisson boundary-control optimization example.
 *
 * The problem identifies an upper-boundary Dirichlet control that matches
 * interior observations of an analytic Poisson state.  It exposes residual,
 * objective, report, and VTU output helpers used by both PETSc and ReSolve
 * example drivers.
 */
class PoissonOptProblem
{
public:
  explicit PoissonOptProblem(const Options& opts);
  ~PoissonOptProblem() = default;

  PoissonOptProblem(const PoissonOptProblem&)            = delete;
  PoissonOptProblem& operator=(const PoissonOptProblem&) = delete;
  PoissonOptProblem(PoissonOptProblem&&)                 = delete;
  PoissonOptProblem& operator=(PoissonOptProblem&&)      = delete;

  const Options&            options() const noexcept;
  const CsrPattern&         statePattern() const;
  const state::Residual&    residual() const;
  const inverse::Objective& objective() const;

  Index numNodes() const noexcept;
  Index numStates() const noexcept;
  Index numParams() const noexcept;
  Index numObservations() const noexcept;

  Report report(const Vector<Real>& prm,
                const Vector<Real>& state,
                Real                value,
                const Vector<Real>& grad) const;

  /**
   * @brief Write final state/control fields to VTU.
   *
   * @param[in] prm - Optimized control vector.
   * @param[in] state - Final state vector.
   * @param[in] output_base - Output path without extension, or with `.vtu`.
   */
  void writeSolution(const Vector<Real>& prm,
                     const Vector<Real>& state,
                     const std::string&  output_base) const;

private:
  void prepareObjective(state::StateSolver& state_solver);

  static Real exactValue(const Mesh::Node& p);

  void initializeBoundaryDofs();
  void initializeTrueControl();
  void initializeObservationLayout();
  void initializeResidual();

  Index        effectiveObservationStride() const;
  Vector<Real> observationWeights() const;

  bool isBoundaryNode(const Mesh::Node& p) const;
  bool isControlNode(const Mesh::Node& p) const;

private:
  Options                     opts_;
  Mesh                        mesh_;
  LagrangeQuadQ1              fe_;
  FESpace                     space_;
  GaussQuadrature             quad_;
  std::unique_ptr<CsrPattern> state_pattern_;

  Vector<Index>  control_dofs_;
  Vector<Index>  fixed_dofs_;
  Vector<Index>  obs_dofs_;
  Vector<Real>   control_weights_;
  Vector<Real>   target_state_;
  Vector<Real>   target_ctr_;
  Vector<Point3> obs_points_;

  std::unique_ptr<assembly::ElementKernel> residual_kernel_;
  std::unique_ptr<state::Residual>         base_residual_;
  std::unique_ptr<state::Residual>         residual_;

  std::unique_ptr<inverse::LeastSquaresObjective> misfit_;
  std::unique_ptr<inverse::LeastSquaresObjective> reg_;
  std::unique_ptr<inverse::SumObjective>          obj_;

  friend Result solve(PoissonOptProblem&    problem,
                      state::Linearization& linearization,
                      linalg::LinearSolver& fwd_lin_solver,
                      linalg::LinearSolver& adj_lin_solver);
};

Result solve(PoissonOptProblem&    problem,
             state::Linearization& linearization,
             linalg::LinearSolver& fwd_lin_solver,
             linalg::LinearSolver& adj_lin_solver);

Options parseOptions(int    argc,
                     char** argv,
                     bool   ignore_unknown);

bool hasPoissonOptHelp(int argc, char** argv);

void printPoissonOptUsage(std::ostream& out,
                          const char*   app_name,
                          bool          petsc_options);

const char* outputDir();

/** @brief Return the problem-specific output file stem. */
std::string outputStem(const Options& opts);

void printReport(std::ostream&            out,
                 const std::string&       backend,
                 const PoissonOptProblem& problem,
                 const Report&            report,
                 Index                    tao_itr,
                 int                      tao_reason);

} // namespace femx::examples::poisson_opt
