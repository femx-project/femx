#pragma once

#include <array>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace io
{

class TimeSeriesDataIn
{
public:
  struct VectorField
  {
    std::string                     name;
    std::array<HostVector<Real>, 3> vals;
  };

  struct Step
  {
    Real                    time{0.0};
    HostVector<VectorField> vecs;
  };

  static TimeSeriesDataIn read(const std::string& path);

  const fem::Mesh& mesh() const;

  Index numSteps() const;

  Real time(Index step) const;

  const std::array<HostVector<Real>, 3>& vectorField(
      Index              step,
      const std::string& name) const;

private:
  fem::Mesh        mesh_;
  HostVector<Step> steps_;
};

} // namespace io
} // namespace femx
