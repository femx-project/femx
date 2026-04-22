#pragma once

#include <array>
#include <vector>

#include "Assembly.hpp"
#include <refem/fe/ElementValues.hpp>
#include <refem/linalg/DenseMatrix.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{

/** 
 * @brief State values evaluated at one quadrature point.
 *
 * Stores velocity, extrapolated advective velocity, velocity gradient,
 * advective derivative, and GLS stabilization parameters.
 */
struct QPState
{
  std::array<real_type, max_dim> u{};
  std::array<real_type, max_dim> u_adv{};
  real_type                      grad_u[max_dim][max_dim]{};
  std::array<real_type, max_dim> u_adv_grad_u{};
  std::array<real_type, max_dim> tau{};
};

/** @brief Assemble the mass terms. */
void assembleMassLHS(const ElementValues& ev,
                          const FluidParams&   fluid,
                          real_type            dt,
                          DenseMatrix&         Ke);

/** @brief Assemble the mass temrs. */
void assembleMassRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          const FluidParams&          fluid,
                          real_type                   dt,
                          Vector&                     Fe);

/** @brief Assemble the advection terms. */
void assembleAdvectionLHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          const FluidParams&          fluid,
                          DenseMatrix&                Ke);

/** @brief Assemble the advection terms. */
void assembleAdvectionRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          const FluidParams&          fluid,
                          Vector&                     Fe);

/** @brief Assemble the diffusion terms. */
void assembleDiffusionLHS(const ElementValues& ev,
                        const FluidParams&   fluid,
                        DenseMatrix&         Ke);

/** @brief Assemble the diffusion terms. */
void assembleDiffusionRHS(const ElementValues&        ev,
                        const std::vector<QPState>& qp_states,
                        const FluidParams&          fluid,
                        Vector&                     Fe);

/** @brief Assemble the pressure-velocity coupling terms. */
void assemblePressureVelocityCouplingLHS(const ElementValues& ev,
                                         DenseMatrix&         Ke);

/** @brief Assemble GLS stabilization terms. */
void assembleStabilizationLHS(const ElementValues&        ev,
                              const std::vector<QPState>& qp_states,
                              const FluidParams&          fluid,
                              real_type                   dt,
                              DenseMatrix&                Ke);

/** @brief Assemble GLS stabilization terms. */
void assembleStabilizationRHS(const ElementValues&        ev,
                              const std::vector<QPState>& qp_states,
                              const FluidParams&          fluid,
                              real_type                   dt,
                              Vector&                     Fe);

} // namespace refem
