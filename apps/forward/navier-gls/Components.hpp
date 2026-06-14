#pragma once

#include <array>
#include <vector>

#include "Config.hpp"
#include <femx/fem/ElementValues.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

/**
 * @brief Values evaluated at one quad point.
 *
 * Stores velocity, extrapolated advective velocity, velocity gradient,
 * advective derivative, and GLS stabilization parameters.
 */
struct QPState
{
  std::array<Real, 3> u{};
  std::array<Real, 3> u_adv{};
  Real                grad_u[3][3]{};
  std::array<Real, 3> u_adv_grad_u{};
  std::array<Real, 3> tau{};
};

/** @brief Assemble the mass terms. */
void assembleMassLHS(const ElementValues& ev,
                     const FluidParams&   fluid,
                     Real                 dt,
                     DenseMatrix&         Ke);

/** @brief Assemble the mass temrs. */
void assembleMassRHS(const ElementValues&        ev,
                     const std::vector<QPState>& qps,
                     const FluidParams&          fluid,
                     Real                        dt,
                     Vector&                     Fe);

/** @brief Assemble the advection terms. */
void assembleAdvectionLHS(const ElementValues&        ev,
                          const std::vector<QPState>& qps,
                          const FluidParams&          fluid,
                          DenseMatrix&                Ke);

/** @brief Assemble the advection terms. */
void assembleAdvectionRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qps,
                          const FluidParams&          fluid,
                          Vector&                     Fe);

/** @brief Assemble the diffusion terms. */
void assembleDiffusionLHS(const ElementValues& ev,
                          const FluidParams&   fluid,
                          DenseMatrix&         Ke);

/** @brief Assemble the diffusion terms. */
void assembleDiffusionRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qps,
                          const FluidParams&          fluid,
                          Vector&                     Fe);

/** @brief Assemble the pressure-velocity coupling terms. */
void assemblePreVelCouplingLHS(const ElementValues& ev,
                               DenseMatrix&         Ke);

/** @brief Assemble GLS stabilization terms. */
void assembleStabilizationLHS(const ElementValues&        ev,
                              const std::vector<QPState>& qps,
                              const FluidParams&          fluid,
                              Real                        dt,
                              DenseMatrix&                Ke);

/** @brief Assemble GLS stabilization terms. */
void assembleStabilizationRHS(const ElementValues&        ev,
                              const std::vector<QPState>& qps,
                              const FluidParams&          fluid,
                              Real                        dt,
                              Vector&                     Fe);

} // namespace femx
