#pragma once

#include <iosfwd>
#include <string>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::examples::poisson
{

struct Options
{
  Index       num_x_cells  = 8;                 ///< Number of cells in x.
  Index       num_y_cells  = 8;                 ///< Number of cells in y.
  MemorySpace backend      = MemorySpace::Host; ///< Selected memory space.
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
class PoissonForwardProblem
{
public:
  explicit PoissonForwardProblem(const Options& opts);

  const Options& options() const noexcept;

  const fem::HostGeometry&         geom() const noexcept;
  const assembly::HostAssemblyMap& map() const noexcept;
  const assembly::HostBoundaryMap& bcMap() const noexcept;
  const HostVector&                bcVals() const noexcept;

  Index numNodes() const noexcept;
  Index numDofs() const noexcept;

  void assemble(HostCsrMatrix& mat, HostVector& rhs) const;

  ErrorReport errorReport(const HostVector& x) const;

  /**
   * @brief Write solution, exact solution, and error fields to VTU.
   *
   * @param[in] x - State vector on this problem's finite-element space.
   * @param[in] base - Output path without extension, or with `.vtu`.
   */
  void writeSolution(const HostVector&  x,
                     const std::string& base) const;

private:
  static Real exactValue(const fem::Mesh::Node& p);
  static Real boundaryValue(const fem::Mesh::Node& p, Real time);
  static bool onBoundary(const fem::Mesh::Node& p, Real time);

private:
  Options                   opts_;
  fem::Mesh                 mesh_;
  fem::LagrangeQuadQ1       fe_;
  fem::FESpace              space_;
  fem::HostGeometry         geom_;
  assembly::HostAssemblyMap map_;
  assembly::HostBoundaryMap bc_map_;
  HostVector                bc_vals_;
};

Options parseOptions(int    argc,
                     char** argv,
                     bool   ignore_unknown);

/** @brief Return the build-local directory for Poisson VTU output. */
const char* outputDir();

/** @brief Return the problem-specific output file stem. */
std::string outputStem(const Options& opts);

void printUsage(const char* app_name,
                bool        petsc_options,
                const char* backend_note = nullptr);

void printReport(std::ostream&                out,
                 const std::string&           backend,
                 const PoissonForwardProblem& problem,
                 const ErrorReport&           error,
                 Real                         res_norm);

} // namespace femx::examples::poisson
