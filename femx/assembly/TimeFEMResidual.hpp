#pragma once

#include <optional>

#include <femx/assembly/TimeElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>
#include <femx/problem/TimeResidual.hpp>

namespace femx
{
namespace assembly
{

/**
 * @brief problem::TimeResidual assembled from elem-local FEM time kernels.
 *
 * TimeFEMResidual gathers local state histories, calls element kernels, and
 * scatters residual or Jacobian contributions into global objects.
 */
class TimeFEMResidual final : public problem::TimeResidual
{
public:
  TimeFEMResidual(Index                    num_steps,
                  DofLayout                res_layout,
                  DofLayout                state_layout,
                  const TimeElementKernel& ker);

  TimeFEMResidual(Index                    num_steps,
                  DofLayout                res_layout,
                  DofLayout                history_state_layout,
                  DofLayout                next_state_layout,
                  const TimeElementKernel& ker);

  TimeFEMResidual(Index                    num_steps,
                  DofLayout                res_layout,
                  Vector<DofLayout>        history_state_layouts,
                  DofLayout                next_state_layout,
                  const TimeElementKernel& ker);

  TimeFEMResidual(Index                    num_steps,
                  DofLayout                res_layout,
                  DofLayout                history_state_layout,
                  DofLayout                next_state_layout,
                  DofLayout                param_layout,
                  const TimeElementKernel& ker);

  TimeFEMResidual(Index                    num_steps,
                  DofLayout                res_layout,
                  Vector<DofLayout>        history_state_layouts,
                  DofLayout                next_state_layout,
                  DofLayout                param_layout,
                  const TimeElementKernel& ker);

  void setElemRange(Index begin, Index end);

  problem::TimeDims dims() const override;

  void res(const problem::TimeContext& ctx,
           Vector<Real>&               out) const override;

  void applyJac(const problem::TimeContext& ctx,
                problem::VariableBlock      wrt,
                const Vector<Real>&         dir,
                Vector<Real>&               out) const override;

  void applyJacT(const problem::TimeContext& ctx,
                 problem::VariableBlock      wrt,
                 const Vector<Real>&         adj,
                 Vector<Real>&               out) const override;

  bool assembleJac(const problem::TimeContext& ctx,
                   problem::VariableBlock      wrt,
                   linalg::MatrixBuilder&      out) const override;

private:
  Index numElems() const;
  Index numParams() const;
  Index numHistoryStates() const;

  const DofLayout* layoutFor(problem::VariableBlock wrt) const;

  void checkLayouts() const;
  void checkContext(const problem::TimeContext& ctx) const;
  void checkVector(const Vector<Real>* value, Index size) const;
  void checkDirection(problem::VariableBlock wrt,
                      const Vector<Real>&    dir) const;

  void gatherHistory(const problem::TimeContext& ctx,
                     Index                       ie,
                     Vector<Real>&               local) const;

  Vector<Real> gatherParam(Index ie, const Vector<Real>& global) const;

  static void gather(const DofLayout&    lyt,
                     const Vector<Real>& global,
                     Index               ie,
                     Vector<Real>&       local);

  static void gather(const DofLayout&       lyt,
                     VectorView<const Real> global,
                     Index                  ie,
                     Vector<Real>&          local);

  static void gather(const DofLayout&       lyt,
                     VectorView<const Real> global,
                     Index                  ie,
                     VectorView<Real>       local);

  static void matVec(const DenseMatrix&  mat,
                     const Vector<Real>& x,
                     Vector<Real>&       out);

  static void matTVec(const DenseMatrix&  mat,
                      const Vector<Real>& x,
                      Vector<Real>&       out);

  static void checkDof(Index id, Index size);

private:
  Index                    num_steps_{0};
  DofLayout                res_layout_;
  Vector<DofLayout>        history_state_layouts_;
  DofLayout                next_state_layout_;
  std::optional<DofLayout> param_layout_;
  const TimeElementKernel& kernel_;
  Index                    elem_begin_{0};
  Index                    elem_end_{0};
};

} // namespace assembly
} // namespace femx
