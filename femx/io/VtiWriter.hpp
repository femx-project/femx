#pragma once

#include <array>
#include <optional>
#include <string>

#include <femx/common/Types.hpp>

namespace femx
{
template <typename T>
class Vector;

namespace io
{

class VtiWriter
{
public:
  struct Image
  {
    std::array<Index, 3> elem_counts = {1, 1, 1};       ///< Cell counts.
    std::array<Real, 3>  origin      = {0.0, 0.0, 0.0}; ///< Grid origin.
    std::array<Real, 3>  spacing     = {1.0, 1.0, 1.0}; ///< Cell spacing.
    std::optional<Real>  time;                          ///< Optional time value.
  };

  struct ElemField
  {
    std::string         name;               ///< VTK field name.
    Index               num_comp = 1; ///< Number of components per cell.
    const Vector<Real>* vals{nullptr};      ///< Field values.
  };

  void writeElemData(const std::string&       fname,
                     const Image&             image,
                     const Vector<ElemField>& fields) const;
};

} // namespace io
} // namespace femx
