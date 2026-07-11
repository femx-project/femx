#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/DofMap.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

/**
 * @brief Scalar or vector-valued finite-element space on one mesh.
 *
 * FESpace combines a mesh, a finite element, and a component count into a
 * global degree-of-freedom layout.  Call setup() after construction and before
 * assembly so element-local dofs can be mapped to global ids.
 */
class FESpace
{
public:
  /**
   * @brief Create a finite-element space on a mesh.
   *
   * @param[in] mesh - Mesh that owns the element topology and coordinates.
   * @param[in] finite_element - Reference finite element used by every cell.
   * @param[in] comps - Number of field components per mesh node.
   */
  FESpace(const Mesh*          mesh,
          const FiniteElement* finite_element,
          Index                comps = 1);

  /**
   * @brief Build the element-to-global-dof map.
   *
   * @pre The mesh and finite element pointers passed to the constructor are
   * valid for the lifetime of this space.
   */
  void setup();

  // Accessors
  const Mesh&          mesh() const noexcept;
  const FiniteElement& finiteElement() const noexcept;
  const DofMap&        dofMap() const noexcept;
  Index                numElems() const noexcept;
  Index                numDofs() const noexcept;
  Index                numComponents() const noexcept;
  Index                numShapesPerElem() const noexcept;
  Index                numDofsPerElem() const noexcept;

  /** @brief Return the local id index for a shape function component. */
  Index localDof(Index shape_index,
                 Index comp) const noexcept;

  /** @brief Return the global id index for a mesh node component. */
  Index globalDof(Index in,
                  Index comp) const noexcept;

  /** @brief Fill the global dof ids used by one element. */
  void elemDofs(Index          ie,
                Vector<Index>& dofs) const;

  /** @brief Return the global dof ids used by one element. */
  Vector<Index> elemDofs(Index ie) const;

private:
  const Mesh*          mesh_{nullptr};
  const FiniteElement* fe_{nullptr};
  DofMap               dof_map_;
  Index                comps_{1};
  Index                num_shapes_per_elem_{0};
  Index                num_dofs_{0};
};

} // namespace fem
} // namespace femx
