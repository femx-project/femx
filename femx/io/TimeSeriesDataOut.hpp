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
    std::string      name;
    HostVector<Real> vals;
  };

  struct VectorField
  {
    std::string                     name;
    std::array<HostVector<Real>, 3> vals;
  };

  struct Step
  {
    Real                    time{0.0};
    HostVector<ScalarField> scalars;
    HostVector<VectorField> vecs;
  };

  void attachMesh(const fem::Mesh& mesh);

  void beginStep(Real time);
  void addNodalScalarField(const std::string&      name,
                           const HostVector<Real>& vals);
  void addNodalVectorField(const std::string&      name,
                           const HostVector<Real>& x,
                           const HostVector<Real>& y);
  void addNodalVectorField(const std::string&      name,
                           const HostVector<Real>& x,
                           const HostVector<Real>& y,
                           const HostVector<Real>& z);
  void clear();

  void write(const std::string& base) const;

private:
  Step& currStep();
  void  checkReady() const;

private:
  const fem::Mesh* mesh_{nullptr};
  HostVector<Step> steps_;
};

} // namespace io
} // namespace femx
