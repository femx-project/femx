#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{

/** @brief Element values on a boundary facet for linear segment/triangle facets. */
class BoundaryElementValues
{
public:
  explicit BoundaryElementValues(const GaussQuadrature& quad);

  void reinit(const Mesh& mesh, const Mesh::BoundaryFacet& facet);

  Index numNodes() const;
  Index dim() const;
  Index numQuadraturePoints() const;

  VectorView<const Real> N(Index iq) const;

  VectorView<const Real> point(Index iq) const;
  VectorView<const Real> nrm(Index iq) const;

  Real point(Index iq, Index comp) const;
  Real nrm(Index iq, Index comp) const;
  Real wt(Index iq) const;
  Real JxW(Index iq) const;

  const Real* NData() const;
  const Real* pointData() const;
  const Real* normalData() const;
  const Real* weightData() const;
  const Real* JxWData() const;

private:
  void reinitSegment(const Mesh& mesh, const Mesh::BoundaryFacet& facet);
  void reinitTriangle(const Mesh& mesh, const Mesh::BoundaryFacet& facet);

  static Element::Shape effectiveShape(const Mesh::BoundaryFacet& facet);

  static Real norm3(Real x, Real y, Real z);

private:
  const GaussQuadrature* quad_{nullptr};

  Index nn_{0};
  Index dim_{0};
  Index nq_{0};

  Vector<Real> N_;
  Vector<Real> pts_;
  Vector<Real> normals_;
  Vector<Real> wts_;
  Vector<Real> JxW_;
};

} // namespace femx
