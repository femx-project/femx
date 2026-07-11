#pragma once

#include <optional>

#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx
{
class DenseMatrix;

template <typename T>
class VectorView;

namespace linalg
{
class MatrixBuilder;
} // namespace linalg

namespace assembly
{

class TimeElementKernel;

/**
 * @brief state::TimeResidual assembled from elem-local FEM time kernels.
 *
 * TimeFEMResidual gathers local state histories, calls element kernels, and
 * scatters residual or Jacobian contributions into global objects.
 */
class TimeFEMResidual final : public state::TimeResidual
{
public:
  TimeFEMResidual(Index                    num_steps,
                  fem::DofLayout           res_layout,
                  fem::DofLayout           state_layout,
                  const TimeElementKernel& ker);

  TimeFEMResidual(Index                    num_steps,
                  fem::DofLayout           res_layout,
                  fem::DofLayout           history_state_layout,
                  fem::DofLayout           next_state_layout,
                  const TimeElementKernel& ker);

  TimeFEMResidual(Index                    num_steps,
                  fem::DofLayout           res_layout,
                  Vector<fem::DofLayout>   history_state_layouts,
                  fem::DofLayout           next_state_layout,
                  const TimeElementKernel& ker);

  TimeFEMResidual(Index                    num_steps,
                  fem::DofLayout           res_layout,
                  fem::DofLayout           history_state_layout,
                  fem::DofLayout           next_state_layout,
                  fem::DofLayout           param_layout,
                  const TimeElementKernel& ker);

  TimeFEMResidual(Index                    num_steps,
                  fem::DofLayout           res_layout,
                  Vector<fem::DofLayout>   history_state_layouts,
                  fem::DofLayout           next_state_layout,
                  fem::DofLayout           param_layout,
                  const TimeElementKernel& ker);

  void setElemRange(Index begin, Index end);

  state::TimeDims dims() const override;

  void res(const state::TimeContext& ctx,
           Vector<Real>&             out) const override;

  void applyJac(const state::TimeContext& ctx,
                state::VariableBlock      wrt,
                const Vector<Real>&       dir,
                Vector<Real>&             out) const override;

  void applyJacT(const state::TimeContext& ctx,
                 state::VariableBlock      wrt,
                 const Vector<Real>&       adj,
                 Vector<Real>&             out) const override;

  bool assembleJac(const state::TimeContext& ctx,
                   state::VariableBlock      wrt,
                   linalg::MatrixBuilder&    out) const override;

private:
  Index numElems() const;
  Index numParams() const;
  Index numHistoryStates() const;

  const fem::DofLayout* layoutFor(state::VariableBlock wrt) const;

  void checkLayouts() const;
  void checkContext(const state::TimeContext& ctx) const;
  void checkVector(const Vector<Real>* value, Index size) const;
  void checkDirection(state::VariableBlock wrt,
                      const Vector<Real>&  dir) const;

  void gatherHistory(const state::TimeContext& ctx,
                     Index                     ie,
                     Vector<Real>&             local) const;

  state::TimeHistoryView historyView(const Vector<Real>& local) const;

  Vector<Real> gatherParam(Index ie, const Vector<Real>& global) const;

  static void gather(const fem::DofLayout& lyt,
                     const Vector<Real>&   global,
                     Index                 ie,
                     Vector<Real>&         local);

  static void gather(const fem::DofLayout&  lyt,
                     VectorView<const Real> global,
                     Index                  ie,
                     Vector<Real>&          local);

  static void gather(const fem::DofLayout&  lyt,
                     VectorView<const Real> global,
                     Index                  ie,
                     VectorView<Real>       local);

  static void checkDof(Index id, Index size);

private:
  Index                         num_steps_{0};
  fem::DofLayout                res_layout_;
  Vector<fem::DofLayout>        history_state_layouts_;
  fem::DofLayout                next_state_layout_;
  std::optional<fem::DofLayout> param_layout_;
  const TimeElementKernel&      kernel_;
  Index                         elem_begin_{0};
  Index                         elem_end_{0};
};

} // namespace assembly
} // namespace femx
