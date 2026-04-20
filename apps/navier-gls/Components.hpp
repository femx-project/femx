#pragma once

#include <array>
#include <vector>

#include "NavierGLS.hpp"

namespace refem
{

class DenseMatrix;
class ElementValues;
class Vector;

struct QPState
{
  std::array<real_type, dim> u{};
  std::array<real_type, dim> u_adv{};
  real_type                  grad_u[dim][dim]{};
  std::array<real_type, dim> u_adv_grad_u{};
  std::array<real_type, dim> tau{};
};

void assembleTransientLHS(const ElementValues& ev,
                          DenseMatrix&         Ke);

void assembleTransientRHS(const ElementValues&         ev,
                          const std::vector<QPState>& qp_states,
                          Vector&                     Fe);

void assembleAdvectionLHS(const ElementValues&         ev,
                          const std::vector<QPState>& qp_states,
                          DenseMatrix&                Ke);

void assembleAdvectionRHS(const ElementValues&         ev,
                          const std::vector<QPState>& qp_states,
                          Vector&                     Fe);

void assembleViscousLHS(const ElementValues& ev,
                        DenseMatrix&         Ke);

void assembleViscousRHS(const ElementValues&         ev,
                        const std::vector<QPState>& qp_states,
                        Vector&                     Fe);

void assemblePressureVelocityCouplingLHS(const ElementValues& ev,
                                         DenseMatrix&         Ke);

void assembleStabilizationLHS(const ElementValues&         ev,
                              const std::vector<QPState>& qp_states,
                              DenseMatrix&                Ke);

void assembleStabilizationRHS(const ElementValues&         ev,
                              const std::vector<QPState>& qp_states,
                              Vector&                     Fe);

} // namespace refem
