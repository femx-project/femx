#pragma once

#include <femx/assembly/BoundaryDofLayout.hpp>
#include <femx/assembly/BoundaryElementKernel.hpp>
#include <femx/assembly/ElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/Residual.hpp>

namespace femx
{
namespace assembly
{

/** @brief problem::Residual assembled from elem-local FEM kernels. */
class FEMResidual final : public problem::Residual
{
public:
  FEMResidual(DofLayout            res_layout,
              DofLayout            state_layout,
              DofLayout            param_layout,
              const ElementKernel& ker);

  FEMResidual(DofLayout            state_layout,
              DofLayout            param_layout,
              const ElementKernel& ker);

  problem::Dimensions dims() const override;

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override;

  void linearize(const Vector<Real>&     state,
                 const Vector<Real>&     prm,
                 problem::Linearization& out) const override;

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
  DofLayout            res_layout_;
  DofLayout            state_layout_;
  DofLayout            param_layout_;
  const ElementKernel& kernel_;
};

/** @brief Adds boundary-facet residual terms to a volume residual. */
class BoundaryFEMResidual final : public problem::Residual
{
public:
  BoundaryFEMResidual(const problem::Residual&     volume,
                      BoundaryDofLayout            res_layout,
                      BoundaryDofLayout            state_layout,
                      BoundaryDofLayout            param_layout,
                      const BoundaryElementKernel& ker);

  problem::Dimensions dims() const override;

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override;

  void linearize(const Vector<Real>&     state,
                 const Vector<Real>&     prm,
                 problem::Linearization& out) const override;

private:
  void addStateJac(const Vector<Real>&    state,
                   const Vector<Real>&    prm,
                   linalg::MatrixBuilder& out) const;

  void addParamJac(const Vector<Real>&    state,
                   const Vector<Real>&    prm,
                   linalg::MatrixBuilder& out) const;

  void checkDimensions() const;

  void checkFacetCompatibility() const;

  void checkGlobalSizes(const Vector<Real>& state,
                        const Vector<Real>& prm,
                        const Vector<Real>& res_out) const;

  static void gather(const BoundaryDofLayout& lyt,
                     const Vector<Real>&      global,
                     Index                    ib,
                     Vector<Real>&            local);

  static void addVec(const BoundaryDofLayout& lyt,
                     Index                    ib,
                     const Vector<Real>&      local,
                     Vector<Real>&            out);

  static void addMat(const BoundaryDofLayout& row_layout,
                     const BoundaryDofLayout& col_layout,
                     Index                    ib,
                     const DenseMatrix&       local,
                     linalg::MatrixBuilder&   out);

  static void checkDof(Index id, Index size);

private:
  const problem::Residual&     volume_;
  BoundaryDofLayout            res_layout_;
  BoundaryDofLayout            state_layout_;
  BoundaryDofLayout            param_layout_;
  const BoundaryElementKernel& kernel_;
};

} // namespace assembly
} // namespace femx
