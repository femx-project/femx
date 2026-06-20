#pragma once

#include <femx/algebra/MatrixBuilder.hpp>
#include <femx/assembly/BoundaryDofLayout.hpp>
#include <femx/assembly/BoundaryElementKernel.hpp>
#include <femx/core/Types.hpp>
#include <femx/problem/MatrixResidualEquation.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace assembly
{

/** @brief Adds boundary-facet residual terms to a volume residual equation. */
class BoundaryResidualEquation final : public problem::MatrixResidualEquation
{
public:
  BoundaryResidualEquation(const problem::MatrixResidualEquation& volume_eq,
                           BoundaryDofLayout                 res_layout,
                           BoundaryDofLayout                 state_layout,
                           BoundaryDofLayout                 param_layout,
                           const BoundaryElementKernel&      kernel);

  Index numStates() const override;

  Index numParams() const override;

  Index numRes() const override;

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override;

  void assembleStateJac(const Vector<Real>&   state,
                        const Vector<Real>&   prm,
                        algebra::SystemMatrix& out) const override;

  void assembleParamJac(const Vector<Real>&   state,
                        const Vector<Real>&   prm,
                        algebra::SystemMatrix& out) const override;

private:
  void checkDimensions() const;

  void checkFacetCompatibility() const;

  void checkGlobalSizes(const Vector<Real>& state,
                        const Vector<Real>& prm,
                        const Vector<Real>& res_out) const;

  static void gather(const BoundaryDofLayout& layout,
                     const Vector<Real>&      global,
                     Index                    ib,
                     Vector<Real>&            local);

  static void addVec(const BoundaryDofLayout& layout,
                     Index                    ib,
                     const Vector<Real>&      local,
                     Vector<Real>&            out);

  static void addMat(const BoundaryDofLayout& row_layout,
                     const BoundaryDofLayout& col_layout,
                     Index                    ib,
                     const DenseMatrix&       local,
                     algebra::MatrixBuilder&  out);

  static void checkDof(Index dof, Index size);

private:
  const problem::MatrixResidualEquation& volume_eq_;
  BoundaryDofLayout                 res_layout_;
  BoundaryDofLayout                 state_layout_;
  BoundaryDofLayout                 param_layout_;
  const BoundaryElementKernel&      kernel_;
};

} // namespace assembly
} // namespace femx
