#pragma once

#include <array>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

namespace fem
{
class Mesh;
} // namespace fem

namespace io
{

class TimeSeriesDataOut
{
public:
  struct ScalarField
  {
    std::string name;
    HostVector  vals;
  };

  struct VectorField
  {
    std::string               name;
    std::array<HostVector, 3> vals;
  };

  struct Step
  {
    Real               time{0.0};
    Array<ScalarField> scalars;
    Array<VectorField> vecs;
  };

  void attachMesh(const fem::Mesh& mesh);

  void beginStep(Real time);
  void addNodalScalarField(const std::string& name,
                           const HostVector&  vals);
  void addNodalVectorField(const std::string& name,
                           const HostVector&  x,
                           const HostVector&  y);
  void addNodalVectorField(const std::string& name,
                           const HostVector&  x,
                           const HostVector&  y,
                           const HostVector&  z);
  void clear();

  void write(const std::string& base) const;

private:
  Step& currStep();
  void  checkReady() const;

private:
  const fem::Mesh* mesh_{nullptr};
  Array<Step>      steps_;
};

} // namespace io
} // namespace femx
