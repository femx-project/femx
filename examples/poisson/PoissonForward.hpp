#pragma once

#include <iosfwd>
#include <string>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ElementQuadratureData.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/runtime/LinearSystemFactory.hpp>
#include <femx/state/Residual.hpp>

namespace femx::examples::poisson
{

/** @brief Command-line configuration for the forward Poisson example. */
struct Options
{
  Index                    num_x_cells = 8; ///< Number of cells in x.
  Index                    num_y_cells = 8; ///< Number of cells in y.
  runtime::ExecutionDevice execution_device =
      runtime::ExecutionDevice::Host; ///< Selected execution device.
  bool write_output = false;          ///< Write VTU output.
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
class PoissonForwardProblem final : public state::HostResidual
{
public:
  /** @brief Construct the mesh, FE space, assembly map, and boundary map. */
  explicit PoissonForwardProblem(const Options& opts);

  /** @brief Return the validated problem options. */
  const Options& options() const noexcept;

  /** @brief Return flattened host geometry. */
  const fem::HostGeometry&              geom() const noexcept;
  /** @brief Return reusable Host element quadrature data. */
  const fem::HostElementQuadratureData& elementData() const noexcept;
  /** @brief Return the element assembly map. */
  const assembly::HostAssemblyMap&      map() const noexcept;
  /** @brief Return essential-boundary CSR metadata. */
  const assembly::HostBoundaryMap&      bcMap() const noexcept;
  /** @brief Return prescribed values in boundary-map order. */
  const HostVector<Real>&               bcVals() const noexcept;

  /** @brief Return the number of mesh nodes. */
  Index numNodes() const noexcept;
  /** @brief Return the number of algebraic unknowns. */
  Index numDofs() const noexcept;

  state::Dimensions     dims() const override;
  const HostCsrPattern& hostPattern() const override;
  void                  res(const HostVector<Real>&             state,
                            const HostVector<Real>&             prm,
                            HostVector<Real>&                   out,
                            linalg::Context<MemorySpace::Host>& ctx) const override;
  void                  assembleStateJac(
                       const HostVector<Real>&              state,
                       const HostVector<Real>&              prm,
                       linalg::Jacobian<MemorySpace::Host>& out,
                       linalg::Context<MemorySpace::Host>&  ctx) const override;
  void applyParamJacT(
      const HostVector<Real>&             state,
      const HostVector<Real>&             prm,
      const HostVector<Real>&             adj,
      HostVector<Real>&                   out,
      linalg::Context<MemorySpace::Host>& ctx) const override;

  /** @brief Compare a solution with the manufactured exact solution. */
  ErrorReport errorReport(const HostVector<Real>& x) const;

  /**
   * @brief Write solution, exact solution, and error fields to VTU.
   *
   * @param[in] x - State vector on this problem's finite-element space.
   * @param[in] base - Output path without extension, or with `.vtu`.
   */
  void writeSolution(const HostVector<Real>& x,
                     const std::string&      base) const;

private:
  void checkVectors(const HostVector<Real>& state,
                    const HostVector<Real>& prm) const;

  static Real exactValue(const fem::Mesh::Node& p);
  static Real boundaryValue(const fem::Mesh::Node& p, Real time);
  static bool onBoundary(const fem::Mesh::Node& p, Real time);

private:
  Options                        opts_;
  fem::Mesh                      mesh_;
  fem::LagrangeQuadQ1            fe_;
  fem::FESpace                   space_;
  fem::HostGeometry              geom_;
  fem::HostElementQuadratureData element_data_;
  assembly::HostAssemblyMap      map_;
  assembly::HostBoundaryMap      bc_map_;
  HostVector<Real>               bc_vals_;
};

/** @brief Parse forward Poisson command-line options. */
Options parseOptions(int argc, char** argv, bool ignore_unknown);

/** @brief Return the build-local directory for Poisson VTU output. */
const char* outputDir();

/** @brief Return the problem-specific output file stem. */
std::string outputStem(const Options& opts);

void printUsage(const char* app_name,
                bool        petsc_options,
                const char* device_note = nullptr);

/** @brief Print the standard forward-solve result summary. */
void printReport(std::ostream&                out,
                 const std::string&           configuration,
                 const PoissonForwardProblem& problem,
                 const ErrorReport&           error,
                 Real                         res_norm);

} // namespace femx::examples::poisson
