#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

class MixedFieldView
{
public:
  /**
   * @brief Create a view of one field inside a mixed finite-element space.
   *
   * @param[in] space - Scalar or vector-valued field space.
   * @param[in] local_offset - Field offset in element-local numbering.
   * @param[in] global_offset - Field offset in global numbering.
   */
  MixedFieldView(const FESpace* space,
                 Index          local_offset,
                 Index          global_offset);

  /** @brief Return the underlying field space. */
  const FESpace& space() const noexcept;
  /** @brief Return the number of field components. */
  Index          numComponents() const noexcept;
  /** @brief Return the number of shapes per element. */
  Index          numShapesPerElem() const noexcept;
  /** @brief Return the number of field degrees of freedom per element. */
  Index          numDofsPerElem() const noexcept;

  /**
   * @brief Return the mixed-space local degree of freedom for a shape component.
   *
   * @param[in] shape_idx - Shape-function index.
   * @param[in] comp - Component index.
   * @return Mixed-space element-local degree-of-freedom index.
   */
  Index localDof(Index shape_idx,
                 Index comp = 0) const noexcept;

  /**
   * @brief Return the mixed-space global degree of freedom.
   *
   * @param[in] scalar_dof - Field-space global degree of freedom.
   * @param[in] comp - Component index.
   * @return Mixed-space global degree-of-freedom index.
   */
  Index globalDof(Index scalar_dof,
                  Index comp = 0) const noexcept;

private:
  const FESpace* space_{nullptr};   ///< Underlying field space.
  Index          local_offset_{0};  ///< Field offset in element-local numbering.
  Index          global_offset_{0}; ///< Field offset in global numbering.
};

class MixedFESpace
{
public:
  /**
   * @brief Add a finite-element field to the mixed space.
   *
   * @param[in] space - Field space.
   */
  void addField(const FESpace& space);

  /** @brief Build offsets and degree-of-freedom maps for all fields. */
  void setup();

  /**
   * @brief Return one mixed-field view.
   *
   * @param[in] fid - Field identifier.
   * @return View of the selected field.
   * @throws std::runtime_error - If the field identifier is out of range.
   */
  MixedFieldView field(Index fid) const;
  /** @brief Return the shared finite-element mesh. */
  const Mesh&    mesh() const noexcept;

  /** @brief Return the element-to-global degree-of-freedom map. */
  const DofMap& dofMap() const noexcept;
  /** @brief Return the number of fields. */
  Index         numFields() const noexcept;
  /** @brief Return the number of elements. */
  Index         numElems() const noexcept;
  /** @brief Return the number of global degrees of freedom. */
  Index         numDofs() const noexcept;
  /** @brief Return the number of mixed degrees of freedom per element. */
  Index         numDofsPerElem() const noexcept;

  /**
   * @brief Fill the mixed-space global degrees of freedom for one element.
   *
   * @param[in] ie - Element index.
   * @param[out] dofs - Global degrees of freedom.
   */
  void elemDofs(Index              ie,
                HostVector<Index>& dofs) const;

  /**
   * @brief Return the mixed-space global degrees of freedom for one element.
   *
   * @param[in] ie - Element index.
   * @return Global degrees of freedom.
   */
  HostVector<Index> elemDofs(Index ie) const;

private:
  HostVector<FESpace> fields_;         ///< Field spaces.
  HostVector<Index>   local_offsets_;  ///< Element-local field offsets.
  HostVector<Index>   global_offsets_; ///< Global field offsets.
  DofMap              dof_map_;        ///< Mixed degree-of-freedom map.
};

} // namespace fem
} // namespace femx
