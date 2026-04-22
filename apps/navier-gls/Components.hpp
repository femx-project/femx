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

/**
 * @brief Assemble the transient contribution to the element matrix.
 *
 * Adds the velocity mass term scaled by fluid density and time step.
 */
void assembleTransientLHS(const ElementValues& ev,
                          const FluidParams&   fluid,
                          real_type            dt,
                          DenseMatrix&         Ke);

/**
 * @brief Assemble the transient contribution to the element residual.
 *
 * Uses the current quadrature-point velocity stored in qp_states.
 */
void assembleTransientRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          const FluidParams&          fluid,
                          real_type                   dt,
                          Vector&                     Fe);

/**
 * @brief Assemble the advective contribution to the element matrix.
 *
 * Uses the extrapolated advective velocity stored in qp_states.
 */
void assembleAdvectionLHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          const FluidParams&          fluid,
                          DenseMatrix&                Ke);

/**
 * @brief Assemble the advective contribution to the element residual.
 *
 * Uses the evaluated advective derivative of velocity.
 */
void assembleAdvectionRHS(const ElementValues&        ev,
                          const std::vector<QPState>& qp_states,
                          const FluidParams&          fluid,
                          Vector&                     Fe);

/**
 * @brief Assemble the viscous contribution to the element matrix.
 *
 * Adds the diffusion-like velocity block using the fluid viscosity.
 */
void assembleViscousLHS(const ElementValues& ev,
                        const FluidParams&   fluid,
                        DenseMatrix&         Ke);

/**
 * @brief Assemble the viscous contribution to the element residual.
 *
 * Uses the velocity gradient stored in qp_states.
 */
void assembleViscousRHS(const ElementValues&        ev,
                        const std::vector<QPState>& qp_states,
                        const FluidParams&          fluid,
                        Vector&                     Fe);

/**
 * @brief Assemble the pressure-velocity coupling terms.
 *
 * Adds the discrete gradient and divergence blocks.
 */
void assemblePressureVelocityCouplingLHS(const ElementValues& ev,
                                         DenseMatrix&         Ke);

/**
 * @brief Assemble GLS stabilization terms for the element matrix.
 *
 * @note Stabilization parameters are taken from qp_states.
 */
void assembleStabilizationLHS(const ElementValues&        ev,
                              const std::vector<QPState>& qp_states,
                              const FluidParams&          fluid,
                              real_type                   dt,
                              DenseMatrix&                Ke);

/**
 * @brief Assemble GLS stabilization terms for the element residual.
 *
 * @note Stabilization parameters are taken from qp_states.
 */
void assembleStabilizationRHS(const ElementValues&        ev,
                              const std::vector<QPState>& qp_states,
                              const FluidParams&          fluid,
                              real_type                   dt,
                              Vector&                     Fe);

} // namespace refem
