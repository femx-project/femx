#pragma once

#include <array>
#include <string>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>

namespace femx
{

class TimeSeriesDataIn
{
public:
  struct VectorField
  {
    std::string                 name;
    std::array<Vector<Real>, 3> values;
  };

  struct Step
  {
    Real                     time{0.0};
    std::vector<VectorField> vecs;
  };

  static TimeSeriesDataIn read(const std::string& path);

  const Mesh& mesh() const;

  Index numSteps() const;

  Real time(Index step) const;

  const std::array<Vector<Real>, 3>& vectorField(
      Index              step,
      const std::string& name) const;

private:
  Mesh              mesh_;
  std::vector<Step> steps_;
};

} // namespace femx
