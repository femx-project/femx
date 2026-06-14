#pragma once

#include <vector>

#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>

namespace femx
{

class MixedFieldView
{
public:
  /** @brief Create a view of one field inside a mixed finite elem space. */
  MixedFieldView(const FESpace* space,
                 Index          local_offset,
                 Index          global_offset);

  // Accessor
  const FESpace& space() const noexcept;
  Index          numComponents() const noexcept;
  Index          numShapesPerElem() const noexcept;
  Index          numDofsPerElem() const noexcept;

  /** @brief Return the mixed-space local dof index for a shape function component. */
  Index localDof(Index shape_index,
                 Index component = 0) const noexcept;

  /** @brief Return the mixed-space global dof index for a scalar dof component. */
  Index globalDof(Index scalar_dof,
                  Index component = 0) const noexcept;

private:
  const FESpace* space_{nullptr};
  Index          local_offset_{0};
  Index          global_offset_{0};
};

class MixedFESpace
{
public:
  /** @brief Add a finite elem field to the mixed space. */
  void addField(const FESpace& space);

  /** @brief Build offsets and dof maps for all fields in the mixed space. */
  void setup();

  // Accessor
  MixedFieldView field(Index field_id) const;
  const Mesh&    mesh() const noexcept;
  Index          numFields() const noexcept;
  Index          numElems() const noexcept;
  Index          numDofs() const noexcept;
  Index          numDofsPerElem() const noexcept;

  /** @brief Fill the mixed-space global dof indices used by one elem. */
  void elemDofs(Index               ic,
                std::vector<Index>& dofs) const;

  /** @brief Return the mixed-space global dof indices used by one elem. */
  std::vector<Index> elemDofs(Index ic) const;

private:
  std::vector<FESpace> fields_;
  std::vector<Index>   local_offsets_;
  std::vector<Index>   global_offsets_;
  Index                num_dofs_per_elem_{0};
  Index                num_dofs_{0};
};

} // namespace femx
