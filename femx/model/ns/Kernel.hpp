#pragma once

#include <femx/assembly/EnzymeTimeVolumeKernel.hpp>
#include <femx/assembly/TimeElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::model::ns
{

void NavierResidual(Index       step,
                    Index       elem,
                    Index       num_qpts,
                    Index       num_nodes,
                    Index       dim,
                    Index       num_residuals,
                    Index       num_history_dofs,
                    Index       num_next_states,
                    Index       num_param,
                    const Real* N,
                    const Real* dNdx,
                    const Real* JxW,
                    const Real* hist,
                    const Real* nxt,
                    const Real* prm,
                    Real*       out);

/**
 * @brief Navier-Stokes element residual and Jacobian kernel.
 *
 * History and variable-parameter Jacobians use Enzyme when available and an
 * element-local central-difference fallback otherwise.
 */
class NavierKernel final : public assembly::TimeElementKernel
{
public:
  NavierKernel(const fem::FESpace&         space,
               const fem::GaussQuadrature& quadrature,
               Index                       num_residuals,
               Index                       num_history_dofs,
               Index                       num_next_states,
               Index                       num_variable_params,
               Vector<Real>                fixed_prm);

  NavierKernel(const fem::FESpace&         space,
               const fem::GaussQuadrature& quadrature,
               Index                       num_residuals,
               Index                       num_history_dofs,
               Index                       num_next_states,
               Vector<Real>                fixed_prm);

  void res(Index                  step,
           Index                  ie,
           state::TimeHistoryView hist,
           const Vector<Real>&    nxt,
           const Vector<Real>&    prm,
           Vector<Real>&          out) const override;

  void jacobian(Index                  step,
                Index                  ie,
                state::VariableBlock   wrt,
                state::TimeHistoryView hist,
                const Vector<Real>&    nxt,
                const Vector<Real>&    prm,
                DenseMatrix&           out) const override;

private:
  void fdJacobian(Index                  step,
                                Index                  ie,
                                state::VariableBlock   wrt,
                                state::TimeHistoryView hist,
                                const Vector<Real>&    nxt,
                                const Vector<Real>&    prm,
                                DenseMatrix&           out) const;

  void checkDimensions();
  void checkInputSizes(state::TimeHistoryView hist,
                       const Vector<Real>&    nxt,
                       const Vector<Real>&    prm) const;

  Vector<Real> makeResidualPrm(const Vector<Real>& variable_prm) const;

private:
  const fem::FESpace&                              space_;
  const fem::GaussQuadrature&                      quad_;
  Index                                            num_residuals_{0};
  Index                                            num_hist_dofs_{0};
  Index                                            num_next_states_{0};
  Index                                            num_hist_states_{1};
  Index                                            num_hist_state_dofs_{0};
  Index                                            num_variable_params_{0};
  Index                                            num_param_{0};
  Vector<Real>                                     fixed_prm_;
  assembly::EnzymeTimeVolumeKernel<NavierResidual> enzyme_kernel_;
};

Vector<Real> physicalParams(Real rho, Real mu, Real dt);

NavierKernel makeNavierKernel(const fem::FESpace&         vel_space,
                              const fem::GaussQuadrature& quadrature,
                              Index                       num_local_dofs,
                              Real                        rho,
                              Real                        mu,
                              Real                        dt);

} // namespace femx::model::ns
