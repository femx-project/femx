#include <cmath>
#include <stdexcept>

#include <femx/fem/BoundaryElementValues.hpp>

using namespace std;

namespace femx
{

BoundaryElementValues::BoundaryElementValues(const GaussQuadrature& quad)
  : quad_(&quad),
    nq_(quad.size())
{
}

void BoundaryElementValues::reinit(const Mesh&                mesh,
                                   const Mesh::BoundaryFacet& facet)
{
  switch (effectiveShape(facet))
  {
  case Element::Shape::Segment:
    reinitSegment(mesh, facet);
    return;

  case Element::Shape::Triangle:
    reinitTriangle(mesh, facet);
    return;

  default:
    throw runtime_error(
        "BoundaryElementValues supports segment and triangle facets only");
  }
}

Index BoundaryElementValues::numNodes() const
{
  return nn_;
}

Index BoundaryElementValues::dim() const
{
  return dim_;
}

Index BoundaryElementValues::numQuadraturePoints() const
{
  return nq_;
}

VectorView<const Real> BoundaryElementValues::N(Index iq) const
{
  return VectorView<const Real>(N_.data() + iq * nn_, nn_);
}

VectorView<const Real> BoundaryElementValues::point(Index iq) const
{
  return VectorView<const Real>(pts_.data() + iq * dim_, dim_);
}

VectorView<const Real> BoundaryElementValues::nrm(Index iq) const
{
  return VectorView<const Real>(
      normals_.data() + iq * dim_, dim_);
}

Real BoundaryElementValues::point(Index iq, Index comp) const
{
  return pts_[iq * dim_ + comp];
}

Real BoundaryElementValues::nrm(Index iq, Index comp) const
{
  return normals_[iq * dim_ + comp];
}

Real BoundaryElementValues::wt(Index iq) const
{
  return wts_[iq];
}

Real BoundaryElementValues::JxW(Index iq) const
{
  return JxW_[iq];
}

const Real* BoundaryElementValues::NData() const
{
  return N_.data();
}

const Real* BoundaryElementValues::pointData() const
{
  return pts_.data();
}

const Real* BoundaryElementValues::normalData() const
{
  return normals_.data();
}

const Real* BoundaryElementValues::weightData() const
{
  return wts_.data();
}

const Real* BoundaryElementValues::JxWData() const
{
  return JxW_.data();
}

void BoundaryElementValues::reinitSegment(const Mesh&                mesh,
                                          const Mesh::BoundaryFacet& facet)
{
  if (quad_->referenceElement() != ReferenceElement::Segment
      || facet.nids.size() != 2)
  {
    throw runtime_error(
        "BoundaryElementValues segment facet requires segment quadrature and two nodes");
  }

  nn_  = 2;
  dim_ = mesh.dim();
  N_.assign(nq_ * nn_, 0.0);
  pts_.assign(nq_ * dim_, 0.0);
  normals_.assign(nq_ * dim_, 0.0);
  wts_.assign(nq_, 0.0);
  JxW_.assign(nq_, 0.0);

  const auto& x0 = mesh.node(facet.nids[0]);
  const auto& x1 = mesh.node(facet.nids[1]);

  const Real tx     = x1[0] - x0[0];
  const Real ty     = x1[1] - x0[1];
  const Real tz     = x1[2] - x0[2];
  const Real length = norm3(tx, ty, tz);
  if (length <= 0.0)
  {
    throw runtime_error("BoundaryElementValues segment has zero length");
  }

  for (Index iq = 0; iq < nq_; ++iq)
  {
    const QuadraturePoint& qp = (*quad_)[iq];
    const Real             n0 = 0.5 * (1.0 - qp[0]);
    const Real             n1 = 0.5 * (1.0 + qp[0]);

    N_[iq * nn_]     = n0;
    N_[iq * nn_ + 1] = n1;

    for (Index d = 0; d < dim_; ++d)
    {
      pts_[iq * dim_ + d] = n0 * x0[d] + n1 * x1[d];
    }

    if (dim_ == 2)
    {
      normals_[iq * dim_]     = ty / length;
      normals_[iq * dim_ + 1] = -tx / length;
    }

    wts_[iq] = qp.wt;
    JxW_[iq] = 0.5 * length * qp.wt;
  }
}

void BoundaryElementValues::reinitTriangle(const Mesh&                mesh,
                                           const Mesh::BoundaryFacet& facet)
{
  if (quad_->referenceElement() != ReferenceElement::Triangle
      || facet.nids.size() != 3 || mesh.dim() != 3)
  {
    throw runtime_error(
        "BoundaryElementValues triangle facet requires triangle quadrature, three nodes, and a 3D mesh");
  }

  nn_  = 3;
  dim_ = mesh.dim();
  N_.assign(nq_ * nn_, 0.0);
  pts_.assign(nq_ * dim_, 0.0);
  normals_.assign(nq_ * dim_, 0.0);
  wts_.assign(nq_, 0.0);
  JxW_.assign(nq_, 0.0);

  const auto& x0 = mesh.node(facet.nids[0]);
  const auto& x1 = mesh.node(facet.nids[1]);
  const auto& x2 = mesh.node(facet.nids[2]);

  const Real e1x = x1[0] - x0[0];
  const Real e1y = x1[1] - x0[1];
  const Real e1z = x1[2] - x0[2];
  const Real e2x = x2[0] - x0[0];
  const Real e2y = x2[1] - x0[1];
  const Real e2z = x2[2] - x0[2];

  const Real nx  = e1y * e2z - e1z * e2y;
  const Real ny  = e1z * e2x - e1x * e2z;
  const Real nz  = e1x * e2y - e1y * e2x;
  const Real J_s = norm3(nx, ny, nz);
  if (J_s <= 0.0)
  {
    throw runtime_error("BoundaryElementValues triangle has zero area");
  }

  for (Index iq = 0; iq < nq_; ++iq)
  {
    const QuadraturePoint& qp = (*quad_)[iq];
    const Real             n0 = 1.0 - qp[0] - qp[1];
    const Real             n1 = qp[0];
    const Real             n2 = qp[1];

    N_[iq * nn_]     = n0;
    N_[iq * nn_ + 1] = n1;
    N_[iq * nn_ + 2] = n2;

    for (Index d = 0; d < dim_; ++d)
    {
      pts_[iq * dim_ + d] = n0 * x0[d] + n1 * x1[d] + n2 * x2[d];
    }

    normals_[iq * dim_]     = nx / J_s;
    normals_[iq * dim_ + 1] = ny / J_s;
    normals_[iq * dim_ + 2] = nz / J_s;

    wts_[iq] = qp.wt;
    JxW_[iq] = J_s * qp.wt;
  }
}

Element::Shape BoundaryElementValues::effectiveShape(
    const Mesh::BoundaryFacet& facet)
{
  if (facet.shape != Element::Shape::Unknown)
  {
    return facet.shape;
  }

  if (facet.nids.size() == 2)
  {
    return Element::Shape::Segment;
  }
  if (facet.nids.size() == 3)
  {
    return Element::Shape::Triangle;
  }

  return Element::Shape::Unknown;
}

Real BoundaryElementValues::norm3(Real x, Real y, Real z)
{
  return sqrt(x * x + y * y + z * z);
}

} // namespace femx
