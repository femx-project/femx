#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/fem/DofMap.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>

namespace femx
{
namespace fem
{

/**
 * @brief Scalar or vector-valued finite-element space on one mesh.
 *
 * FESpace combines a mesh, a finite element, and a component count into a
 * global degree-of-freedom layout. Call setup() after construction and before
 * assembly so element-local degrees of freedom can be mapped to global
 * identifiers.
 */
class FESpace
{
public:
  /**
   * @brief Create a finite-element space on a mesh.
   *
   * @param[in] mesh - Mesh that owns the element topology and coordinates.
   * @param[in] fe - Reference finite element used by every cell.
   * @param[in] comps - Number of field components per mesh node.
   */
  FESpace(const Mesh*          mesh,
          const FiniteElement* fe,
          Index                comps = 1);

  /**
   * @brief Build the element-to-global-dof map.
   *
   * @pre The mesh and finite element pointers passed to the constructor are
   * valid for the lifetime of this space.
   */
  void setup();

  /**
   * @brief Return the finite-element mesh.
   */
  const Mesh&          mesh() const noexcept;
  /**
   * @brief Return the reference finite element.
   */
  const FiniteElement& finiteElement() const noexcept;
  /**
   * @brief Return the element-to-global degree-of-freedom map.
   */
  const DofMap&        dofMap() const noexcept;
  /**
   * @brief Return the number of elements.
   */
  Index                numElems() const noexcept;
  /**
   * @brief Return the number of degrees of freedom.
   */
  Index                numDofs() const noexcept;
  /**
   * @brief Return the number of field components.
   */
  Index                numComponents() const noexcept;
  /**
   * @brief Return the number of shapes per element.
   */
  Index                numShapesPerElem() const noexcept;
  /**
   * @brief Return the number of degrees of freedom per element.
   */
  Index                numDofsPerElem() const noexcept;

  /**
   * @brief Return the local degree of freedom for a shape component.
   *
   * @param[in] shape_idx - Shape-function index.
   * @param[in] comp - Component index.
   * @return Element-local degree-of-freedom index.
   */
  Index localDof(Index shape_idx,
                 Index comp) const noexcept;

  /**
   * @brief Return the global degree of freedom for a node component.
   *
   * @param[in] in - Mesh-node index.
   * @param[in] comp - Component index.
   * @return Global degree-of-freedom index.
   */
  Index globalDof(Index in,
                  Index comp) const noexcept;

  /**
   * @brief Fill the global degrees of freedom used by one element.
   *
   * @param[in]  ie - Element index.
   * @param[out] dofs - Global degrees of freedom.
   */
  void elemDofs(Index              ie,
                HostVector<Index>& dofs) const;

  /**
   * @brief Return the global degrees of freedom used by one element.
   *
   * @param[in] ie - Element index.
   * @return Global degrees of freedom.
   */
  HostVector<Index> elemDofs(Index ie) const;

private:
  const Mesh*          mesh_{nullptr};          ///< Finite-element mesh.
  const FiniteElement* fe_{nullptr};            ///< Reference finite element.
  DofMap               dof_map_;                ///< Degree-of-freedom map.
  Index                comps_{1};               ///< Number of field components.
  Index                num_shapes_per_elem_{0}; ///< Number of shapes per element.
};

} // namespace fem
} // namespace femx
