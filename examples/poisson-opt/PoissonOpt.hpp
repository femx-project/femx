#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/inverse/LeastSquaresObjective.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/inverse/SumObjective.hpp>
#include <femx/linalg/Backend.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::examples::poisson_opt
{

/** @brief Command-line configuration for Poisson boundary optimization. */
struct Options
{
  Index num_x_cells  = 32;     ///< Number of cells in x.
  Index num_y_cells  = 32;     ///< Number of cells in y.
  bool  write_output = false;  ///< Write VTU output.
  Real  alpha        = 1.0e-6; ///< Control regularization weight.
  Index obs_stride   = 0;      ///< Observation spacing in mesh cells.
  Index max_its      = 50;     ///< Maximum TAO iterations.
};

/**
 * @brief Final metrics reported by the Poisson optimization example.
 */
struct Report
{
  Real value         = 0.0; ///< Final objective value.
  Real grad_norm     = 0.0; ///< Final reduced-gradient norm.
  Real state_rms_err = 0.0; ///< RMS state error against the target.
  Real state_max_err = 0.0; ///< Maximum state error against the target.
  Real ctr_rms_err   = 0.0; ///< RMS control error against the target.
  Real ctr_max_err   = 0.0; ///< Maximum control error against the target.
};

/**
 * @brief Optimizer result together with final control and state vectors.
 */
struct Result
{
  Report     report;             ///< Final diagnostic metrics.
  HostVector prm;                ///< Optimized control vector.
  HostVector state;              ///< State vector at the optimized control.
  Index      tao_itr    = 0;     ///< Number of TAO iterations.
  int        tao_reason = 0;     ///< PETSc/TAO convergence reason.
  bool       converged  = false; ///< True when TAO reports convergence.
};

/**
 * @brief Poisson boundary-control optimization example.
 */
class PoissonOptProblem
{
public:
  /** @brief Construct all discretization and objective data. */
  explicit PoissonOptProblem(const Options& opts);
  ~PoissonOptProblem() = default;

  PoissonOptProblem(const PoissonOptProblem&)            = delete;
  PoissonOptProblem& operator=(const PoissonOptProblem&) = delete;
  PoissonOptProblem(PoissonOptProblem&&)                 = delete;
  PoissonOptProblem& operator=(PoissonOptProblem&&)      = delete;

  /** @brief Return the validated optimization options. */
  const Options&                   options() const noexcept;
  /** @brief Return the state Jacobian assembly map. */
  const assembly::HostAssemblyMap& stateMap() const;
  /** @brief Return the reduced objective components. */
  const inverse::Objective&        objective() const;

  /** @brief Return the number of mesh nodes. */
  Index numNodes() const noexcept;
  /** @brief Return the state-vector size. */
  Index numStates() const noexcept;
  /** @brief Return the boundary-control size. */
  Index numParams() const noexcept;
  /** @brief Return the number of observation samples. */
  Index numObservations() const noexcept;

  /** @brief Compute final error and gradient metrics. */
  Report report(const HostVector& prm,
                const HostVector& state,
                Real              value,
                const HostVector& grad) const;

  /**
   * @brief Write final state/control fields to VTU.
   *
   * @param[in] prm - Optimized control vector.
   * @param[in] state - Final state vector.
   * @param[in] base - Output path without extension, or with `.vtu`.
   */
  void writeSolution(const HostVector&  prm,
                     const HostVector&  state,
                     const std::string& base) const;

private:
  void prepareObjective(HostVector target_state);

  static Real exactValue(const fem::Mesh::Node& p);

  void initializeBoundaryDofs();
  void initializeTrueControl();
  void initializeObservationLayout();
  void writeFields(const HostVector&  prm,
                   const HostVector&  state,
                   const std::string& path) const;

  void writeObs(const HostVector&  state,
                const std::string& path) const;

  Index      effectiveObservationStride() const;
  HostVector observationWeights() const;

  bool isBoundaryNode(const fem::Mesh::Node& p) const;
  bool isControlNode(const fem::Mesh::Node& p) const;

private:
  Options                   opts_;
  fem::Mesh                 mesh_;
  fem::LagrangeQuadQ1       fe_;
  fem::FESpace              space_;
  fem::HostGeometry         geom_;
  assembly::HostAssemblyMap state_map_;

  Array<Index>  ctr_dofs_;
  Array<Index>  fixed_dofs_;
  Array<Index>  obs_dofs_;
  HostVector    ctr_weights_;
  HostVector    target_state_;
  HostVector    target_ctr_;
  Array<Point3> obs_points_;

  std::unique_ptr<inverse::LeastSquaresObjective> misfit_;
  std::unique_ptr<inverse::LeastSquaresObjective> reg_;
  std::unique_ptr<inverse::SumObjective>          obj_;

  template <class Backend>
  friend Result solve(PoissonOptProblem&             problem,
                      typename Backend::Mat&         fwd_jac,
                      linalg::LinearSolver<Backend>& fwd_solver,
                      typename Backend::Mat&         adj_jac,
                      linalg::LinearSolver<Backend>& adj_solver,
                      typename Backend::Ctx&         ctx);
};

/** @brief Optimize boundary values using the supplied forward/adjoint solvers. */
template <class Backend>
Result solve(PoissonOptProblem&             problem,
             typename Backend::Mat&         fwd_jac,
             linalg::LinearSolver<Backend>& fwd_solver,
             typename Backend::Mat&         adj_jac,
             linalg::LinearSolver<Backend>& adj_solver,
             typename Backend::Ctx&         ctx);

/** @brief Parse Poisson optimization command-line options. */
Options parseOptions(int argc, char** argv, bool ignore_unknown);

/** @brief Print command-line usage for an optimization executable. */
void printUsage(std::ostream& out,
                const char*   app_name,
                bool          petsc_options);

/** @brief Return the build-local directory for optimization output. */
const char* outputDir();

/** @brief Return the problem-specific output file stem. */
std::string outputStem(const Options& opts);

/** @brief Print the standard optimization result summary. */
void printReport(std::ostream&            out,
                 const std::string&       backend,
                 const PoissonOptProblem& problem,
                 const Report&            report,
                 Index                    tao_itr,
                 int                      tao_reason);

} // namespace femx::examples::poisson_opt
