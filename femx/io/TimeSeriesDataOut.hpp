#pragma once

#include <array>
#include <string>
#include <vector>

#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{

class Mesh;

class TimeSeriesDataOut
{
public:
  struct ScalarField
  {
    std::string  name;
    Vector<Real> values;
  };

  struct VectorField
  {
    std::string                 name;
    std::array<Vector<Real>, 3> values;
  };

  struct Step
  {
    Real                     time{0.0};
    std::vector<ScalarField> scalars;
    std::vector<VectorField> vecs;
  };

  void attachMesh(const Mesh& mesh);

  void beginStep(Real time);
  void addNodalScalarField(const std::string&  name,
                           const Vector<Real>& values);
  void addNodalVectorField(const std::string&  name,
                           const Vector<Real>& x,
                           const Vector<Real>& y);
  void addNodalVectorField(const std::string&  name,
                           const Vector<Real>& x,
                           const Vector<Real>& y,
                           const Vector<Real>& z);
  void clear();

  void write(const std::string& basename) const;

private:
  Step& currentStep();
  void  checkReady() const;

private:
  const Mesh*       mesh_{nullptr};
  std::vector<Step> steps_;
};

} // namespace femx
