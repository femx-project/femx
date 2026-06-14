#pragma once

#include <functional>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/mesh/Mesh.hpp>

namespace femx
{

class MixedFieldView;
class FESpace;
class SparseMatrix;
class Vector;

/** @brief Stores and applies Dirichlet boundary constraints.
 *
 * Constraints can be added directly by degree of freedom, by physical boundary
 * tag, or by a user-provided boundary marker.
 */
class DirichletCondition
{
public:
  /** @brief Marks whether a mesh node belongs to the constrained boundary. */
  using BoundaryMarker = std::function<bool(const Mesh::Node&, Real)>;

  /** @brief Evaluates the prescribed boundary value at a mesh node and time. */
  using BoundaryValue = std::function<Real(const Mesh::Node&, Real)>;

  DirichletCondition() = default;

  /** @brief Add one constrained degree of freedom. */
  void addDof(Index dof, Real value);

  /** @brief Add a constant value on a physical boundary tag. */
  void addBoundary(const FESpace& space,
                   Index          physical_tag,
                   Real           value,
                   Real           time      = 0.0,
                   Index          component = 0);

  /** @brief Add a value function on a physical boundary tag. */
  void addBoundary(const FESpace&       space,
                   Index                physical_tag,
                   const BoundaryValue& value,
                   Real                 time      = 0.0,
                   Index                component = 0);

  /** @brief Add a constant value on a mixed field physical boundary tag. */
  void addBoundary(const MixedFieldView& field,
                   Index                 physical_tag,
                   Real                  value,
                   Real                  time      = 0.0,
                   Index                 component = 0);

  /** @brief Add a value function on a mixed field physical boundary tag. */
  void addBoundary(const MixedFieldView& field,
                   Index                 physical_tag,
                   const BoundaryValue&  value,
                   Real                  time      = 0.0,
                   Index                 component = 0);

  /** @brief Add a constant value on nodes selected by a boundary marker. */
  void addBoundary(const FESpace&        space,
                   const BoundaryMarker& marker,
                   Real                  value,
                   Real                  time      = 0.0,
                   Index                 component = 0);

  /** @brief Add a value function on nodes selected by a boundary marker. */
  void addBoundary(const FESpace&        space,
                   const BoundaryMarker& marker,
                   const BoundaryValue&  value,
                   Real                  time      = 0.0,
                   Index                 component = 0);

  /** @brief Add a constant value on mixed field nodes selected by a marker. */
  void addBoundary(const MixedFieldView& field,
                   const BoundaryMarker& marker,
                   Real                  value,
                   Real                  time      = 0.0,
                   Index                 component = 0);

  /** @brief Add a value function on mixed field nodes selected by a marker. */
  void addBoundary(const MixedFieldView& field,
                   const BoundaryMarker& marker,
                   const BoundaryValue&  value,
                   Real                  time      = 0.0,
                   Index                 component = 0);

  const std::vector<Index>& dofs() const noexcept;
  const std::vector<Real>&  values() const noexcept;

  /** @brief Apply the constraints to a matrix and right-hand side. */
  void apply(SparseMatrix& A, Vector& b) const;

private:
  std::vector<Index> dofs_;
  std::vector<Real>  values_;
};

} // namespace femx
