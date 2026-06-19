#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

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
    std::array<Index, 3> cell_counts = {1, 1, 1};
    std::array<Real, 3>  origin      = {0.0, 0.0, 0.0};
    std::array<Real, 3>  spacing     = {1.0, 1.0, 1.0};
    std::optional<Real>  time;
  };

  struct CellField
  {
    std::string         name;
    Index               num_components = 1;
    const Vector<Real>* values{nullptr};
  };

  void writeCellData(const std::string&             filename,
                     const Image&                   image,
                     const std::vector<CellField>&  fields) const;
};

} // namespace femx
