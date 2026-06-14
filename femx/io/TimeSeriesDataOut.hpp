#pragma once

#include <array>
#include <string>
#include <vector>

#include <femx/core/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class Mesh;

class TimeSeriesDataOut
{
public:
  struct ScalarField
  {
    std::string name;
    Vector      values;
  };

  struct VectorField
  {
    std::string           name;
    std::array<Vector, 3> values;
  };

  struct Step
  {
    real_type                time{0.0};
    std::vector<ScalarField> scalars;
    std::vector<VectorField> vectors;
  };

  void attachMesh(const Mesh& mesh);

  void beginStep(real_type time);
  void addNodalScalarField(const std::string& name,
                           const Vector&      values);
  void addNodalVectorField(const std::string& name,
                           const Vector&      x,
                           const Vector&      y);
  void addNodalVectorField(const std::string& name,
                           const Vector&      x,
                           const Vector&      y,
                           const Vector&      z);
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
