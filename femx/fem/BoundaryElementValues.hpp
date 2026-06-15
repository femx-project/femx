#pragma once

#include <vector>

#include <femx/common/Types.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/linalg/VectorView.hpp>
#include <femx/mesh/Mesh.hpp>

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
  VectorView<const Real> normal(Index iq) const;

  Real point(Index iq, Index component) const;
  Real normal(Index iq, Index component) const;
  Real weight(Index iq) const;
  Real JxW(Index iq) const;

  const Real* NData() const;
  const Real* pointData() const;
  const Real* normalData() const;
  const Real* weightData() const;
  const Real* JxWData() const;

private:
  void reinitSegment(const Mesh& mesh, const Mesh::BoundaryFacet& facet);
  void reinitTriangle(const Mesh& mesh, const Mesh::BoundaryFacet& facet);

  static Cell::Shape effectiveShape(const Mesh::BoundaryFacet& facet);

  static Real norm3(Real x, Real y, Real z);

private:
  const GaussQuadrature* quad_{nullptr};

  Index num_nodes_{0};
  Index dim_{0};
  Index num_qp_{0};

  std::vector<Real> N_;
  std::vector<Real> points_;
  std::vector<Real> normals_;
  std::vector<Real> weights_;
  std::vector<Real> JxW_;
};

} // namespace femx
