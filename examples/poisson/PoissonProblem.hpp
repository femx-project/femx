#pragma once

#include <iosfwd>
#include <string>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/fem/ElementQuadData.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>

namespace femx::examples::poisson
{

/**
 * @brief Command-line configuration for the forward Poisson example.
 */
struct Options
{
  Index       num_x_cells  = 8;                 ///< Number of cells in x.
  Index       num_y_cells  = 8;                 ///< Number of cells in y.
  MemorySpace memspace     = MemorySpace::Host; ///< Selected memory space.
  bool        write_output = false;             ///< Write VTU output.
};

/**
 * @brief Error and range metrics reported by the forward Poisson example.
 */
struct ErrorReport
{
  Real min_val = 0.0; ///< Minimum numerical solution value.
  Real max_val = 0.0; ///< Maximum numerical solution value.
  Real rms_err = 0.0; ///< RMS nodal error against the exact solution.
  Real max_err = 0.0; ///< Maximum nodal error against the exact solution.
};

/**
 * @brief Forward Poisson example on a structured quadrilateral mesh.
 */
class PoissonProblem
{
public:
  /**
   * @brief Construct the mesh, FE space, assembly map, and boundary map.
   *
   * @param[in] opts - Validated problem options.
   */
  explicit PoissonProblem(const Options& opts);

  /**
   * @brief Return the validated problem options.
   */
  const Options& options() const noexcept;

  /**
   * @brief Return the finite-element mesh.
   */
  const fem::Mesh& mesh() const noexcept;

  /**
   * @brief Return reusable Host element quadrature data.
   */
  const fem::HostElementQuadData& elementData() const noexcept;

  /**
   * @brief Return the element assembly map.
   */
  const assembly::HostAssemblyMap& assemblyMap() const noexcept;

  /**
   * @brief Return essential-boundary CSR metadata.
   */
  const assembly::HostBoundaryMap& boundaryMap() const noexcept;

  /**
   * @brief Return prescribed values in boundary-map order.
   */
  const HostVector<Real>& boundaryValues() const noexcept;

  /**
   * @brief Return the number of mesh nodes.
   */
  Index numNodes() const noexcept;
  /**
   * @brief Return the number of algebraic unknowns.
   */
  Index numDofs() const noexcept;

  /**
   * @brief Compare a solution with the manufactured exact solution.
   *
   * @param[in] x - State vector.
   * @return Error and solution-range metrics.
   */
  ErrorReport errorReport(const HostVector<Real>& x) const;

  /**
   * @brief Write solution, exact solution, and error fields to VTU.
   *
   * @param[in] x    - State vector on this problem's finite-element space.
   * @param[in] base - Output path without extension, or with `.vtu`.
   */
  void writeSolution(const HostVector<Real>& x,
                     const std::string&      base) const;

private:
  static Real exactValue(const fem::Mesh::Node& p);
  static Real boundaryValue(const fem::Mesh::Node& p, Real time);
  static bool onBoundary(const fem::Mesh::Node& p, Real time);

  Options                   opts_;            ///< Problem options.
  fem::Mesh                 mesh_;            ///< Finite-element mesh.
  fem::LagrangeQuadQ1       fe_;              ///< Scalar finite element.
  fem::FESpace              space_;           ///< Scalar finite-element space.
  fem::HostElementQuadData  elem_data_;       ///< Host integration data.
  assembly::HostAssemblyMap assm_map_;        ///< Host assembly mapping.
  assembly::HostBoundaryMap boundary_map_;    ///< Constrained rows.
  HostVector<Real>          boundary_values_; ///< Prescribed boundary values.
};

/**
 * @brief Parse forward Poisson command-line options.
 *
 * @param[in] argc           - Argument count.
 * @param[in] argv           - Argument values.
 * @param[in] ignore_unknown - Whether to ignore solver-owned options.
 * @return Parsed and validated options.
 * @throws std::runtime_error If validation fails.
 */
Options parseOptions(int argc, char** argv, bool ignore_unknown);

/**
 * @brief Return the build-local directory for Poisson VTU output.
 */
const char* outputDir();

/**
 * @brief Return the problem-specific output file stem.
 *
 * @param[in] opts - Problem options.
 * @return Output file stem.
 */
std::string outputStem(const Options& opts);

/**
 * @brief Print command-line usage for the forward Poisson example.
 *
 * @param[in] app_name      - Executable name.
 * @param[in] petsc_options - Whether PETSc options are accepted.
 * @param[in] backend_note  - Optional execution-backend note.
 */
void printUsage(const char* app_name,
                bool        petsc_options,
                const char* backend_note = nullptr);

/**
 * @brief Print the standard forward-solve result summary.
 *
 * @param[in,out] out           - Output stream.
 * @param[in]     configuration - Solver configuration name.
 * @param[in]     problem       - Solved problem.
 * @param[in]     err           - Solution error metrics.
 * @param[in]     rnorm         - Residual L2 norm.
 */
void printReport(std::ostream&         out,
                 const std::string&    configuration,
                 const PoissonProblem& problem,
                 const ErrorReport&    err,
                 Real                  rnorm);

} // namespace femx::examples::poisson
