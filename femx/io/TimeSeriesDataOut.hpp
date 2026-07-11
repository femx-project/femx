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
    std::string  name;
    Vector<Real> vals;
  };

  struct VectorField
  {
    std::string                 name;
    std::array<Vector<Real>, 3> vals;
  };

  struct Step
  {
    Real                time{0.0};
    Vector<ScalarField> scalars;
    Vector<VectorField> vecs;
  };

  void attachMesh(const fem::Mesh& mesh);

  void beginStep(Real time);
  void addNodalScalarField(const std::string&  name,
                           const Vector<Real>& vals);
  void addNodalVectorField(const std::string&  name,
                           const Vector<Real>& x,
                           const Vector<Real>& y);
  void addNodalVectorField(const std::string&  name,
                           const Vector<Real>& x,
                           const Vector<Real>& y,
                           const Vector<Real>& z);
  void clear();

  void write(const std::string& base) const;

private:
  Step& currentStep();
  void  checkReady() const;

private:
  const fem::Mesh* mesh_{nullptr};
  Vector<Step>     steps_;
};

} // namespace io
} // namespace femx
