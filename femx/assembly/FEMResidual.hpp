#pragma once

#include <optional>

#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/state/Residual.hpp>

namespace femx
{
template <typename T>
class Vector;

namespace linalg
{
class MatrixBuilder;
} // namespace linalg

namespace assembly
{

class ElementKernel;

/**
 * @brief state::Residual assembled from element-local FEM kernels.
 *
 * FEMResidual gathers global state/parameter values to each element, evaluates
 * an ElementKernel, and scatters residual and Jacobian contributions back into
 * global vectors and matrices.
 */
class FEMResidual final : public state::Residual
{
public:
  FEMResidual(DofLayout            res_layout,
              DofLayout            state_layout,
              DofLayout            param_layout,
              const ElementKernel& ker);

  FEMResidual(DofLayout            state_layout,
              DofLayout            param_layout,
              const ElementKernel& ker);

  FEMResidual(DofLayout            state_layout,
              const ElementKernel& ker);

  state::Dimensions dims() const override;

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override;

  void linearize(const Vector<Real>&   state,
                 const Vector<Real>&   prm,
                 state::Linearization& out) const override;

private:
  Index numElems() const;

  void assembleStateJac(const Vector<Real>&    state,
                        const Vector<Real>&    prm,
                        linalg::MatrixBuilder& out) const;

  void assembleParamJac(const Vector<Real>&    state,
                        const Vector<Real>&    prm,
                        linalg::MatrixBuilder& out) const;

  void checkElemCounts() const;

  void checkGlobalSizes(const Vector<Real>& state,
                        const Vector<Real>& prm) const;

  static void gather(const DofLayout&    lyt,
                     const Vector<Real>& global,
                     Index               ie,
                     Vector<Real>&       local);

private:
  DofLayout                res_layout_;
  DofLayout                state_layout_;
  std::optional<DofLayout> param_layout_;
  const ElementKernel&     kernel_;
};

} // namespace assembly
} // namespace femx
