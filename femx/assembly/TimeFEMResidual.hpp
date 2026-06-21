#pragma once

#include <optional>

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/assembly/TimeElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/problem/TimeResidual.hpp>

namespace femx
{
namespace assembly
{

/** @brief problem::TimeResidual assembled from cell-local FEM time kernels. */
class TimeFEMResidual final : public problem::TimeResidual
{
public:
  TimeFEMResidual(Index                    num_steps,
                  DofLayout                res_layout,
                  DofLayout                state_layout,
                  const TimeElementKernel& kernel);

  TimeFEMResidual(Index                    num_steps,
                  DofLayout                res_layout,
                  DofLayout                prev_state_layout,
                  DofLayout                next_state_layout,
                  const TimeElementKernel& kernel);

  TimeFEMResidual(Index                    num_steps,
                  DofLayout                res_layout,
                  DofLayout                prev_state_layout,
                  DofLayout                next_state_layout,
                  DofLayout                param_layout,
                  const TimeElementKernel& kernel);

  void setCellRange(Index begin, Index end);

  problem::TimeDims dimensions() const override;

  void residual(const problem::TimeContext& ctx,
                Vector<Real>&               out) const override;

  void applyJac(const problem::TimeContext& ctx,
                     problem::VariableBlock      wrt,
                     const Vector<Real>&         dir,
                     Vector<Real>&               out) const override;

  void applyJacT(const problem::TimeContext& ctx,
                      problem::VariableBlock      wrt,
                      const Vector<Real>&         adjoint,
                      Vector<Real>&               out) const override;

  bool assembleJacobian(const problem::TimeContext& ctx,
                        problem::VariableBlock      wrt,
                        linalg::MatrixBuilder&     out) const override;

private:
  Index numCells() const;
  Index numParams() const;

  const DofLayout* layoutFor(problem::VariableBlock wrt) const;

  void checkLayouts() const;
  void checkContext(const problem::TimeContext& ctx) const;
  void checkVector(const Vector<Real>* value, Index size, const char* name) const;
  void checkDirection(problem::VariableBlock wrt,
                      const Vector<Real>&    dir) const;

  Vector<Real> gatherParam(Index ic, const Vector<Real>& global) const;

  static void gather(const DofLayout&    layout,
                     const Vector<Real>& global,
                     Index               ic,
                     Vector<Real>&       local);

  static void matVec(const DenseMatrix&  mat,
                     const Vector<Real>& x,
                     Vector<Real>&       out);

  static void matTVec(const DenseMatrix&  mat,
                      const Vector<Real>& x,
                      Vector<Real>&       out);

  static void resize(Vector<Real>& out, Index size);
  static void checkDof(Index dof, Index size);

private:
  Index                    num_steps_{0};
  DofLayout                res_layout_;
  DofLayout                prev_state_layout_;
  DofLayout                next_state_layout_;
  std::optional<DofLayout> param_layout_;
  const TimeElementKernel& kernel_;
  Index                    cell_begin_{0};
  Index                    cell_end_{0};
};

} // namespace assembly
} // namespace femx
