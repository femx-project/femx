#pragma once

#include <femx/assembly/HostTimeResidual.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::model::ns
{

/**
 * @brief Navier-Stokes element residual and Jacobian kernel.
 *
 * History and variable-parameter Jacobians use Enzyme when available and an
 * element-local central-difference fallback otherwise.
 */
class NavierKernel final : public assembly::HostElementOperator
{
public:
  NavierKernel(const fem::FESpace&         space,
               const fem::GaussQuadrature& quad,
               Index                       num_res,
               Index                       num_hist_dofs,
               Index                       num_next,
               Index                       num_var_prm,
               HostVector                  fixed_prm);

  NavierKernel(const fem::FESpace&         space,
               const fem::GaussQuadrature& quad,
               Index                       num_res,
               Index                       num_hist_dofs,
               Index                       num_next,
               HostVector                  fixed_prm);

  void res(Index                  step,
           Index                  ie,
           state::TimeHistoryView hist,
           const HostVector&      nxt,
           const HostVector&      prm,
           HostVector&            out) const override;

  void jac(Index                  step,
           Index                  ie,
           state::VariableBlock   wrt,
           state::TimeHistoryView hist,
           const HostVector&      nxt,
           const HostVector&      prm,
           DenseMatrix&           out) const override;

private:
  void fdJac(Index                  step,
             Index                  ie,
             state::VariableBlock   wrt,
             state::TimeHistoryView hist,
             const HostVector&      nxt,
             const HostVector&      prm,
             DenseMatrix&           out) const;

#if defined(FEMX_HAS_ENZYME)
  void adJac(Index                  step,
             Index                  ie,
             state::VariableBlock   wrt,
             state::TimeHistoryView hist,
             const HostVector&      nxt,
             const HostVector&      prm,
             DenseMatrix&           out) const;
#endif

  void checkDims();
  void checkSizes(state::TimeHistoryView hist,
                  const HostVector&      nxt,
                  const HostVector&      prm) const;

  HostVector makeResPrm(const HostVector& var_prm) const;

private:
  const fem::FESpace&         space_;
  const fem::GaussQuadrature& quad_;
  Index                       num_res_{0};
  Index                       num_hist_dofs_{0};
  Index                       num_next_{0};
  Index                       num_hist_{1};
  Index                       num_hist_dof_{0};
  Index                       num_var_prm_{0};
  Index                       num_prm_{0};
  HostVector                  fixed_prm_;
};

HostVector physicalParams(Real rho, Real mu, Real dt);

NavierKernel makeNavierKernel(const fem::FESpace&         vel_sp,
                              const fem::GaussQuadrature& quad,
                              Index                       ndof,
                              Real                        rho,
                              Real                        mu,
                              Real                        dt);

} // namespace femx::model::ns
