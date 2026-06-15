#include <cmath>
#include <stdexcept>

#include <femx/fem/BoundaryElementValues.hpp>

namespace femx
{

BoundaryElementValues::BoundaryElementValues(const GaussQuadrature& quad)
  : quad_(&quad),
    num_qp_(quad.size())
{
}

void BoundaryElementValues::reinit(const Mesh&                mesh,
                                   const Mesh::BoundaryFacet& facet)
{
  switch (effectiveShape(facet))
  {
  case Cell::Shape::Segment:
    reinitSegment(mesh, facet);
    return;

  case Cell::Shape::Triangle:
    reinitTriangle(mesh, facet);
    return;

  default:
    throw std::runtime_error(
        "BoundaryElementValues supports segment and triangle facets only");
  }
}

Index BoundaryElementValues::numNodes() const
{
  return num_nodes_;
}

Index BoundaryElementValues::dim() const
{
  return dim_;
}

Index BoundaryElementValues::numQuadraturePoints() const
{
  return num_qp_;
}

VectorView<const Real> BoundaryElementValues::N(Index iq) const
{
  return VectorView<const Real>(
      N_.data() + static_cast<std::size_t>(iq * num_nodes_), num_nodes_);
}

VectorView<const Real> BoundaryElementValues::point(Index iq) const
{
  return VectorView<const Real>(
      points_.data() + static_cast<std::size_t>(iq * dim_), dim_);
}

VectorView<const Real> BoundaryElementValues::normal(Index iq) const
{
  return VectorView<const Real>(
      normals_.data() + static_cast<std::size_t>(iq * dim_), dim_);
}

Real BoundaryElementValues::point(Index iq, Index component) const
{
  return points_[static_cast<std::size_t>(iq * dim_ + component)];
}

Real BoundaryElementValues::normal(Index iq, Index component) const
{
  return normals_[static_cast<std::size_t>(iq * dim_ + component)];
}

Real BoundaryElementValues::weight(Index iq) const
{
  return weights_[static_cast<std::size_t>(iq)];
}

Real BoundaryElementValues::JxW(Index iq) const
{
  return JxW_[static_cast<std::size_t>(iq)];
}

const Real* BoundaryElementValues::NData() const
{
  return N_.data();
}

const Real* BoundaryElementValues::pointData() const
{
  return points_.data();
}

const Real* BoundaryElementValues::normalData() const
{
  return normals_.data();
}

const Real* BoundaryElementValues::weightData() const
{
  return weights_.data();
}

const Real* BoundaryElementValues::JxWData() const
{
  return JxW_.data();
}

void BoundaryElementValues::reinitSegment(const Mesh&                mesh,
                                          const Mesh::BoundaryFacet& facet)
{
  if (quad_->referenceElement() != ReferenceElement::Segment
      || facet.node_ids.size() != 2)
  {
    throw std::runtime_error(
        "BoundaryElementValues segment facet requires segment quadrature and two nodes");
  }

  num_nodes_ = 2;
  dim_       = mesh.dim();
  N_.assign(static_cast<std::size_t>(num_qp_ * num_nodes_), 0.0);
  points_.assign(static_cast<std::size_t>(num_qp_ * dim_), 0.0);
  normals_.assign(static_cast<std::size_t>(num_qp_ * dim_), 0.0);
  weights_.assign(static_cast<std::size_t>(num_qp_), 0.0);
  JxW_.assign(static_cast<std::size_t>(num_qp_), 0.0);

  const auto& x0 = mesh.node(facet.node_ids[0]);
  const auto& x1 = mesh.node(facet.node_ids[1]);

  const Real tx     = x1[0] - x0[0];
  const Real ty     = x1[1] - x0[1];
  const Real tz     = x1[2] - x0[2];
  const Real length = norm3(tx, ty, tz);
  if (length <= 0.0)
  {
    throw std::runtime_error("BoundaryElementValues segment has zero length");
  }

  for (Index iq = 0; iq < num_qp_; ++iq)
  {
    const QuadraturePoint& qp = (*quad_)[iq];
    const Real             n0 = 0.5 * (1.0 - qp[0]);
    const Real             n1 = 0.5 * (1.0 + qp[0]);

    N_[static_cast<std::size_t>(iq * num_nodes_)]     = n0;
    N_[static_cast<std::size_t>(iq * num_nodes_ + 1)] = n1;

    for (Index d = 0; d < dim_; ++d)
    {
      points_[static_cast<std::size_t>(iq * dim_ + d)] =
          n0 * x0[d] + n1 * x1[d];
    }

    if (dim_ == 2)
    {
      normals_[static_cast<std::size_t>(iq * dim_)] =
          ty / length;
      normals_[static_cast<std::size_t>(iq * dim_ + 1)] =
          -tx / length;
    }

    weights_[static_cast<std::size_t>(iq)] = qp.weight;
    JxW_[static_cast<std::size_t>(iq)]     = 0.5 * length * qp.weight;
  }
}

void BoundaryElementValues::reinitTriangle(const Mesh&                mesh,
                                           const Mesh::BoundaryFacet& facet)
{
  if (quad_->referenceElement() != ReferenceElement::Triangle
      || facet.node_ids.size() != 3 || mesh.dim() != 3)
  {
    throw std::runtime_error(
        "BoundaryElementValues triangle facet requires triangle quadrature, three nodes, and a 3D mesh");
  }

  num_nodes_ = 3;
  dim_       = mesh.dim();
  N_.assign(static_cast<std::size_t>(num_qp_ * num_nodes_), 0.0);
  points_.assign(static_cast<std::size_t>(num_qp_ * dim_), 0.0);
  normals_.assign(static_cast<std::size_t>(num_qp_ * dim_), 0.0);
  weights_.assign(static_cast<std::size_t>(num_qp_), 0.0);
  JxW_.assign(static_cast<std::size_t>(num_qp_), 0.0);

  const auto& x0 = mesh.node(facet.node_ids[0]);
  const auto& x1 = mesh.node(facet.node_ids[1]);
  const auto& x2 = mesh.node(facet.node_ids[2]);

  const Real e1x = x1[0] - x0[0];
  const Real e1y = x1[1] - x0[1];
  const Real e1z = x1[2] - x0[2];
  const Real e2x = x2[0] - x0[0];
  const Real e2y = x2[1] - x0[1];
  const Real e2z = x2[2] - x0[2];

  const Real nx    = e1y * e2z - e1z * e2y;
  const Real ny    = e1z * e2x - e1x * e2z;
  const Real nz    = e1x * e2y - e1y * e2x;
  const Real jac_s = norm3(nx, ny, nz);
  if (jac_s <= 0.0)
  {
    throw std::runtime_error("BoundaryElementValues triangle has zero area");
  }

  for (Index iq = 0; iq < num_qp_; ++iq)
  {
    const QuadraturePoint& qp = (*quad_)[iq];
    const Real             n0 = 1.0 - qp[0] - qp[1];
    const Real             n1 = qp[0];
    const Real             n2 = qp[1];

    N_[static_cast<std::size_t>(iq * num_nodes_)]     = n0;
    N_[static_cast<std::size_t>(iq * num_nodes_ + 1)] = n1;
    N_[static_cast<std::size_t>(iq * num_nodes_ + 2)] = n2;

    for (Index d = 0; d < dim_; ++d)
    {
      points_[static_cast<std::size_t>(iq * dim_ + d)] =
          n0 * x0[d] + n1 * x1[d] + n2 * x2[d];
    }

    normals_[static_cast<std::size_t>(iq * dim_)]     = nx / jac_s;
    normals_[static_cast<std::size_t>(iq * dim_ + 1)] = ny / jac_s;
    normals_[static_cast<std::size_t>(iq * dim_ + 2)] = nz / jac_s;

    weights_[static_cast<std::size_t>(iq)] = qp.weight;
    JxW_[static_cast<std::size_t>(iq)]     = jac_s * qp.weight;
  }
}

Cell::Shape BoundaryElementValues::effectiveShape(
    const Mesh::BoundaryFacet& facet)
{
  if (facet.shape != Cell::Shape::Unknown)
  {
    return facet.shape;
  }

  if (facet.node_ids.size() == 2)
  {
    return Cell::Shape::Segment;
  }
  if (facet.node_ids.size() == 3)
  {
    return Cell::Shape::Triangle;
  }

  return Cell::Shape::Unknown;
}

Real BoundaryElementValues::norm3(Real x, Real y, Real z)
{
  return std::sqrt(x * x + y * y + z * z);
}

} // namespace femx
