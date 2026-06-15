#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>

namespace femx
{
namespace assembly
{

/** @brief Boundary-facet-local residual and Jacobian kernel. */
class BoundaryElementKernel
{
public:
  virtual ~BoundaryElementKernel() = default;

  virtual void res(Index                       ib,
                   const Mesh::BoundaryFacet& facet,
                   const Vector&              u,
                   const Vector&              m,
                   Vector&                    out) const = 0;

  virtual void stateJac(Index                       ib,
                        const Mesh::BoundaryFacet& facet,
                        const Vector&              u,
                        const Vector&              m,
                        DenseMatrix&               out) const = 0;

  virtual void paramJac(Index                       ib,
                        const Mesh::BoundaryFacet& facet,
                        const Vector&              u,
                        const Vector&              m,
                        DenseMatrix&               out) const = 0;
};

} // namespace assembly
} // namespace femx
