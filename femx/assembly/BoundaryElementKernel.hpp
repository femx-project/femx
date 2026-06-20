#pragma once

#include <femx/core/Types.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/fem/Mesh.hpp>

namespace femx
{
namespace assembly
{

/** @brief Boundary-facet-local residual and Jacobian kernel. */
class BoundaryElementKernel
{
public:
  virtual ~BoundaryElementKernel() = default;

  virtual void res(Index                      ib,
                   const Mesh::BoundaryFacet& facet,
                   const Vector<Real>&        u,
                   const Vector<Real>&        m,
                   Vector<Real>&              out) const = 0;

  virtual void stateJac(Index                      ib,
                        const Mesh::BoundaryFacet& facet,
                        const Vector<Real>&        u,
                        const Vector<Real>&        m,
                        DenseMatrix&               out) const = 0;

  virtual void paramJac(Index                      ib,
                        const Mesh::BoundaryFacet& facet,
                        const Vector<Real>&        u,
                        const Vector<Real>&        m,
                        DenseMatrix&               out) const = 0;
};

} // namespace assembly
} // namespace femx
