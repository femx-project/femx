#pragma once

#include <array>
#include <optional>
#include <string>

#include <femx/common/Types.hpp>

namespace femx
{

template <typename T>
class Vector;

class VtiWriter
{
public:
  struct Image
  {
    std::array<Index, 3> elem_counts = {1, 1, 1};
    std::array<Real, 3>  origin      = {0.0, 0.0, 0.0};
    std::array<Real, 3>  spacing     = {1.0, 1.0, 1.0};
    std::optional<Real>  time;
  };

  struct ElemField
  {
    std::string         name;
    Index               num_components = 1;
    const Vector<Real>* vals{nullptr};
  };

  void writeElemData(const std::string&       fname,
                     const Image&             image,
                     const Vector<ElemField>& fields) const;
};

} // namespace femx
