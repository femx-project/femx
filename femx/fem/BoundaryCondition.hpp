#pragma once

#include <functional>

#include <femx/common/Types.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class MixedFieldView;
class FESpace;
class CsrMatrix;

/**
 * @brief Stores and applies Dirichlet boundary constraints.
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
  void addDof(Index id, Real value);

  /** @brief Add a constant value on a physical boundary tag. */
  void addBoundary(const FESpace& space,
                   Index          ptag,
                   Real           value,
                   Real           time = 0.0,
                   Index          comp = 0);

  /** @brief Add a value function on a physical boundary tag. */
  void addBoundary(const FESpace&       space,
                   Index                ptag,
                   const BoundaryValue& value,
                   Real                 time = 0.0,
                   Index                comp = 0);

  /** @brief Add a constant value on a mixed field physical boundary tag. */
  void addBoundary(const MixedFieldView& field,
                   Index                 ptag,
                   Real                  value,
                   Real                  time = 0.0,
                   Index                 comp = 0);

  /** @brief Add a value function on a mixed field physical boundary tag. */
  void addBoundary(const MixedFieldView& field,
                   Index                 ptag,
                   const BoundaryValue&  value,
                   Real                  time = 0.0,
                   Index                 comp = 0);

  /** @brief Add a constant value on nodes selected by a boundary marker. */
  void addBoundary(const FESpace&        space,
                   const BoundaryMarker& mark,
                   Real                  value,
                   Real                  time = 0.0,
                   Index                 comp = 0);

  /** @brief Add a value function on nodes selected by a boundary marker. */
  void addBoundary(const FESpace&        space,
                   const BoundaryMarker& mark,
                   const BoundaryValue&  value,
                   Real                  time = 0.0,
                   Index                 comp = 0);

  /** @brief Add a constant value on mixed field nodes selected by a marker. */
  void addBoundary(const MixedFieldView& field,
                   const BoundaryMarker& mark,
                   Real                  value,
                   Real                  time = 0.0,
                   Index                 comp = 0);

  /** @brief Add a value function on mixed field nodes selected by a marker. */
  void addBoundary(const MixedFieldView& field,
                   const BoundaryMarker& mark,
                   const BoundaryValue&  value,
                   Real                  time = 0.0,
                   Index                 comp = 0);

  const Vector<Index>& dofs() const noexcept;
  const Vector<Real>&  vals() const noexcept;

  /** @brief Apply the constraints to a matrix and right-hand side. */
  void apply(CsrMatrix& A, Vector<Real>& b) const;

private:
  Vector<Index> dofs_;
  Vector<Real>  vals_;
};

using BoundaryCondition = DirichletCondition;

} // namespace femx
