#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/common/Workspace.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace linalg
{
class CsrAssemblyMatrix;
}
} // namespace femx

namespace femx::examples::poisson
{

struct Options
{
  Index         num_x_cells  = 32;
  Index         num_y_cells  = 32;
  WorkspaceType backend      = WorkspaceType::Cpu;
  bool          write_output = false;
};

/**
 * @brief Error and range metrics reported by the forward Poisson example.
 *
 * ErrorReport summarizes solution extrema and pointwise error against the
 * analytic reference solution.
 */
struct ErrorReport
{
  Real min_value = 0.0;
  Real max_value = 0.0;
  Real rms_err   = 0.0;
  Real max_err   = 0.0;
};

/**
 * @brief Forward Poisson example on a structured quadrilateral mesh.
 *
 * The problem solves a scalar Laplace equation with analytic boundary data,
 * reports nodal error against the exact solution, and can write VTU
 * visualization fields.
 */
class PoissonForwardProblem
{
public:
  explicit PoissonForwardProblem(const Options& opts);

  const Options&    options() const noexcept;
  const CsrPattern& pattern() const;

  Index numNodes() const noexcept;
  Index numDofs() const noexcept;

  void assemble(linalg::CsrAssemblyMatrix& A,
                Vector<Real>&              rhs) const;

  ErrorReport errorReport(const Vector<Real>& x) const;

  /**
   * @brief Write solution, exact solution, and error fields to VTU.
   *
   * @param[in] x - State vector on this problem's finite-element space.
   * @param[in] output_base - Output path without extension, or with `.vtu`.
   */
  void writeSolution(const Vector<Real>& x,
                     const std::string&  output_base) const;

private:
  static Real exactValue(const Mesh::Node& p);
  static Real boundaryValue(const Mesh::Node& p, Real time);
  static bool onBoundary(const Mesh::Node& p, Real time);

private:
  Options                     opts_;
  Mesh                        mesh_;
  LagrangeQuadQ1              fe_;
  FESpace                     space_;
  GaussQuadrature             quad_;
  std::unique_ptr<CsrPattern> pattern_;
};

Options parseOptions(int    argc,
                     char** argv,
                     bool   ignore_unknown);

/** @brief Return the build-local directory for Poisson VTU output. */
const char* outputDir();

/** @brief Return the problem-specific output file stem. */
std::string outputStem(const Options& opts);

void printUsage(const char* app_name,
                bool        petsc_options);

void printReport(std::ostream&                out,
                 const std::string&           backend,
                 const PoissonForwardProblem& problem,
                 const ErrorReport&           error,
                 Real                         residual_norm);

} // namespace femx::examples::poisson
