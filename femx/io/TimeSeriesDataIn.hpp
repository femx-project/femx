#pragma once

#include <array>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class TimeSeriesDataIn
{
public:
  struct VectorField
  {
    std::string                 name;
    std::array<Vector<Real>, 3> vals;
  };

  struct Step
  {
    Real                time{0.0};
    Vector<VectorField> vecs;
  };

  static TimeSeriesDataIn read(const std::string& path);

  const Mesh& mesh() const;

  Index numSteps() const;

  Real time(Index step) const;

  const std::array<Vector<Real>, 3>& vectorField(
      Index              step,
      const std::string& name) const;

private:
  Mesh         mesh_;
  Vector<Step> steps_;
};

} // namespace femx
