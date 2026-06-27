#pragma once

#include <femx/assembly/EnzymeTimeVolumeKernel.hpp>
#include <femx/assembly/TimeElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::navier
{

void NavierResidual(Index       step,
                    Index       elem,
                    Index       nq,
                    Index       nn,
                    Index       dim,
                    Index       nres,
                    Index       num_prev_states,
                    Index       num_next_states,
                    Index       nprm,
                    const Real* N,
                    const Real* dNdx,
                    const Real* JxW,
                    const Real* prev,
                    const Real* nxt,
                    const Real* prm,
                    Real*       out);

class NavierKernel final : public assembly::TimeElementKernel
{
public:
  NavierKernel(const FESpace&         space,
               const GaussQuadrature& quadrature,
               Index                  nres,
               Index                  num_prev_states,
               Index                  num_next_states,
               Index                  num_variable_prm,
               Vector<Real>           fixed_prm);

  NavierKernel(const FESpace&         space,
               const GaussQuadrature& quadrature,
               Index                  nres,
               Index                  num_prev_states,
               Index                  num_next_states,
               Vector<Real>           fixed_prm);

  void res(Index                    step,
           Index                    ie,
           problem::TimeHistoryView hist,
           const Vector<Real>&      nxt,
           const Vector<Real>&      prm,
           Vector<Real>&            out) const override;

  void jacobian(Index                    step,
                Index                    ie,
                problem::VariableBlock   wrt,
                problem::TimeHistoryView hist,
                const Vector<Real>&      nxt,
                const Vector<Real>&      prm,
                DenseMatrix&             out) const override;

private:
  void checkDimensions();
  void checkInputSizes(problem::TimeHistoryView hist,
                       const Vector<Real>&      nxt,
                       const Vector<Real>&      prm) const;

  Vector<Real> makeResidualPrm(const Vector<Real>& variable_prm) const;

private:
  const FESpace&                                   space_;
  const GaussQuadrature&                           quad_;
  Index                                            nres_{0};
  Index                                            num_prev_states_{0};
  Index                                            num_next_states_{0};
  Index                                            num_hist_states_{1};
  Index                                            num_hist_state_dofs_{0};
  Index                                            num_variable_prm_{0};
  Index                                            nprm_{0};
  Vector<Real>                                     fixed_prm_;
  assembly::EnzymeTimeVolumeKernel<NavierResidual> fallback_;
};

Vector<Real> physicalParams(Real rho, Real mu, Real dt);

NavierKernel makeNavierKernel(const FESpace&         vel_space,
                              const GaussQuadrature& quadrature,
                              Index                  nloc,
                              Real                   rho,
                              Real                   mu,
                              Real                   dt);

} // namespace femx::navier
