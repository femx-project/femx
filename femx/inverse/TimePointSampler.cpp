#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <femx/fem/FiniteElement.hpp>
#include <femx/linalg/MatrixView.hpp>
#include <femx/linalg/VectorView.hpp>
#include <femx/mesh/Cell.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/inverse/TimePointSampler.hpp>

namespace femx
{
namespace inverse
{

namespace
{

constexpr Real point_tol = 1.0e-10;
constexpr Real det_tol   = 1.0e-14;

struct ScalarStencil
{
  std::vector<Index> node_ids;
  Vector<Real>       weights;
};

bool insideBox(const Cell&   cell,
               const Point3& point,
               Index         dim)
{
  for (Index a = 0; a < dim; ++a)
  {
    Real lower = cell.node(0)[a];
    Real upper = cell.node(0)[a];
    for (Index in = 1; in < cell.numNodes(); ++in)
    {
      lower = std::min(lower, cell.node(in)[a]);
      upper = std::max(upper, cell.node(in)[a]);
    }
    if (point[a] < lower - point_tol || point[a] > upper + point_tol)
    {
      return false;
    }
  }
  return true;
}

bool insideSimplex(const Vector<Real>& weights)
{
  Real sum = 0.0;
  for (Real weight : weights)
  {
    if (weight < -point_tol || weight > 1.0 + point_tol)
    {
      return false;
    }
    sum += weight;
  }
  return std::abs(sum - 1.0) <= 10.0 * point_tol;
}

Vector<Real> shapeWeights(const FiniteElement& fe,
                          const QuadraturePoint& qp)
{
  Vector<Real> weights(fe.numDofsPerElement());
  fe.calcShape(qp, VectorView<Real>(weights.data(), weights.size()));
  return weights;
}

bool triWeights(const FiniteElement& fe,
                const Cell&          cell,
                const Point3&        point,
                Vector<Real>&        weights)
{
  const Point3 a   = cell.node(0);
  const Point3 e1  = difference(cell.node(1), a);
  const Point3 e2  = difference(cell.node(2), a);
  const Point3 rhs = difference(point, a);

  const Real det = e1[0] * e2[1] - e1[1] * e2[0];
  if (std::abs(det) < det_tol)
  {
    return false;
  }

  const Real r = (rhs[0] * e2[1] - rhs[1] * e2[0]) / det;
  const Real s = (e1[0] * rhs[1] - e1[1] * rhs[0]) / det;

  weights = shapeWeights(fe, QuadraturePoint{{r, s, 0.0}, 0.0});
  return insideSimplex(weights);
}

bool tetWeights(const FiniteElement& fe,
                const Cell&          cell,
                const Point3&        point,
                Vector<Real>&        weights)
{
  const Point3 a   = cell.node(0);
  const Point3 e1  = difference(cell.node(1), a);
  const Point3 e2  = difference(cell.node(2), a);
  const Point3 e3  = difference(cell.node(3), a);
  const Point3 rhs = difference(point, a);

  const Real det = dot(e1, cross(e2, e3));
  if (std::abs(det) < det_tol)
  {
    return false;
  }

  const Real r = dot(rhs, cross(e2, e3)) / det;
  const Real s = dot(e1, cross(rhs, e3)) / det;
  const Real t = dot(e1, cross(e2, rhs)) / det;

  weights = shapeWeights(fe, QuadraturePoint{{r, s, t}, 0.0});
  return insideSimplex(weights);
}

Point3 mappedPoint(const Cell&         cell,
                   const Vector<Real>& weights,
                   Index               dim)
{
  Point3 mapped{0.0, 0.0, 0.0};
  for (Index in = 0; in < cell.numNodes(); ++in)
  {
    for (Index a = 0; a < dim; ++a)
    {
      mapped[a] += weights[in] * cell.node(in)[a];
    }
  }
  return mapped;
}

bool quadSolveStep(const Cell&          cell,
                   const FiniteElement& fe,
                   Real                 r,
                   Real                 s,
                   const Point3&        point,
                   Real&                dr,
                   Real&                ds,
                   Vector<Real>&        weights)
{
  const QuadraturePoint qp{{r, s, 0.0}, 0.0};
  weights = shapeWeights(fe, qp);

  std::vector<Real> grad(
      static_cast<std::size_t>(fe.numDofsPerElement() * fe.dim()));
  fe.calcShapeGrad(
      qp,
      MatrixView<Real>(
          grad.data(), fe.numDofsPerElement(), fe.dim()));

  Real j00 = 0.0;
  Real j01 = 0.0;
  Real j10 = 0.0;
  Real j11 = 0.0;
  for (Index in = 0; in < cell.numNodes(); ++in)
  {
    const Real x = cell.node(in)[0];
    const Real y = cell.node(in)[1];
    j00 += x * grad[static_cast<std::size_t>(in * fe.dim())];
    j01 += x * grad[static_cast<std::size_t>(in * fe.dim() + 1)];
    j10 += y * grad[static_cast<std::size_t>(in * fe.dim())];
    j11 += y * grad[static_cast<std::size_t>(in * fe.dim() + 1)];
  }

  const Point3 phys = mappedPoint(cell, weights, fe.dim());
  const Real   res0 = phys[0] - point[0];
  const Real   res1 = phys[1] - point[1];
  const Real   det  = j00 * j11 - j01 * j10;
  if (std::abs(det) < det_tol)
  {
    return false;
  }

  dr = (j11 * res0 - j01 * res1) / det;
  ds = (-j10 * res0 + j00 * res1) / det;
  return true;
}

bool quadWeights(const FiniteElement& fe,
                 const Cell&          cell,
                 const Point3&        point,
                 Vector<Real>&        weights)
{
  Real r = 0.0;
  Real s = 0.0;

  for (Index iter = 0; iter < 12; ++iter)
  {
    Real dr = 0.0;
    Real ds = 0.0;
    if (!quadSolveStep(cell, fe, r, s, point, dr, ds, weights))
    {
      return false;
    }

    r -= dr;
    s -= ds;

    if (std::abs(dr) + std::abs(ds) <= point_tol)
    {
      break;
    }
  }

  weights = shapeWeights(fe, QuadraturePoint{{r, s, 0.0}, 0.0});
  const Point3 phys = mappedPoint(cell, weights, fe.dim());
  const Real   err0 = phys[0] - point[0];
  const Real   err1 = phys[1] - point[1];
  const bool   inside =
      r >= -1.0 - point_tol && r <= 1.0 + point_tol
      && s >= -1.0 - point_tol && s <= 1.0 + point_tol;
  return inside
         && (err0 * err0 + err1 * err1 <= 100.0 * point_tol * point_tol);
}

bool cellWeights(const FiniteElement& fe,
                 const Cell&          cell,
                 const Point3&        point,
                 Vector<Real>&        weights)
{
  switch (fe.referenceElement())
  {
  case ReferenceElement::Triangle:
    return triWeights(fe, cell, point, weights);

  case ReferenceElement::Quadrilateral:
    return quadWeights(fe, cell, point, weights);

  case ReferenceElement::Tetrahedron:
    return tetWeights(fe, cell, point, weights);

  case ReferenceElement::Segment:
    break;
  }

  throw std::runtime_error(
      "TimePointSampler does not support this reference element");
}

ScalarStencil findScalarStencil(const FESpace& space,
                                const Point3&  point)
{
  const Mesh&          mesh = space.mesh();
  const FiniteElement& fe   = space.finiteElement();
  if (mesh.dim() != fe.dim())
  {
    throw std::runtime_error(
        "TimePointSampler mesh dimension does not match finite element");
  }

  for (Index ic = 0; ic < mesh.numElems(); ++ic)
  {
    const Cell& cell = mesh.cell(ic);
    if (cell.numNodes() != fe.numNodes())
    {
      throw std::runtime_error(
          "TimePointSampler cell node count does not match finite element");
    }
    if (!insideBox(cell, point, mesh.dim()))
    {
      continue;
    }

    Vector<Real> weights;
    if (cellWeights(fe, cell, point, weights))
    {
      return ScalarStencil{cell.nodeIds(), weights};
    }
  }

  throw std::runtime_error("TimePointSampler point is outside the mesh");
}

} // namespace

TimePointSampler::TimePointSampler(Index               num_steps,
                                   const MixedFESpace& space,
                                   Index               field_id,
                                   std::vector<Point3> points,
                                   Vector<Index>       components,
                                   Index               num_prm)
  : num_steps_(num_steps),
    num_states_(space.numDofs()),
    num_prm_(num_prm),
    points_(std::move(points)),
    components_(std::move(components))
{
  if (num_steps_ < 0 || num_states_ < 0 || num_prm_ < 0)
  {
    throw std::runtime_error("TimePointSampler received invalid dimensions");
  }

  const MixedFieldView field = space.field(field_id);
  if (components_.empty())
  {
    for (Index c = 0; c < field.numComponents(); ++c)
    {
      components_.push_back(c);
    }
  }

  for (Index component : components_)
  {
    if (component < 0 || component >= field.numComponents())
    {
      throw std::runtime_error(
          "TimePointSampler component is out of range");
    }
  }

  stencils_ = buildStencils(field, points_, components_);
}

Index TimePointSampler::numSteps() const
{
  return num_steps_;
}

Index TimePointSampler::numStates() const
{
  return num_states_;
}

Index TimePointSampler::numParams() const
{
  return num_prm_;
}

Index TimePointSampler::numObservations() const
{
  return static_cast<Index>(stencils_.size());
}

void TimePointSampler::observe(Index               level,
                               const Vector<Real>& state,
                               const Vector<Real>& prm,
                               Vector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  resize(out, numObservations());

  for (Index i = 0; i < numObservations(); ++i)
  {
    const Stencil& stencil = stencils_[static_cast<std::size_t>(i)];
    for (Index j = 0; j < stencil.indices.size(); ++j)
    {
      out[i] += stencil.weights[j] * state[stencil.indices[j]];
    }
  }
}

void TimePointSampler::applyStateJac(Index               level,
                                     const Vector<Real>& state,
                                     const Vector<Real>& prm,
                                     const Vector<Real>& dir,
                                     Vector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numStates())
  {
    throw std::runtime_error(
        "TimePointSampler state direction size mismatch");
  }

  resize(out, numObservations());
  for (Index i = 0; i < numObservations(); ++i)
  {
    const Stencil& stencil = stencils_[static_cast<std::size_t>(i)];
    for (Index j = 0; j < stencil.indices.size(); ++j)
    {
      out[i] += stencil.weights[j] * dir[stencil.indices[j]];
    }
  }
}

void TimePointSampler::applyStateJacT(Index               level,
                                      const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      const Vector<Real>& dir,
                                      Vector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numObservations())
  {
    throw std::runtime_error(
        "TimePointSampler observation direction size mismatch");
  }

  resize(out, numStates());
  for (Index i = 0; i < numObservations(); ++i)
  {
    const Stencil& stencil = stencils_[static_cast<std::size_t>(i)];
    for (Index j = 0; j < stencil.indices.size(); ++j)
    {
      out[stencil.indices[j]] += stencil.weights[j] * dir[i];
    }
  }
}

void TimePointSampler::applyParamJac(Index               level,
                                     const Vector<Real>& state,
                                     const Vector<Real>& prm,
                                     const Vector<Real>& dir,
                                     Vector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numParams())
  {
    throw std::runtime_error(
        "TimePointSampler parameter direction size mismatch");
  }

  resize(out, numObservations());
}

void TimePointSampler::applyParamJacT(Index               level,
                                      const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      const Vector<Real>& dir,
                                      Vector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numObservations())
  {
    throw std::runtime_error(
        "TimePointSampler observation direction size mismatch");
  }

  resize(out, numParams());
}

const std::vector<Point3>& TimePointSampler::points() const
{
  return points_;
}

const Vector<Index>& TimePointSampler::components() const
{
  return components_;
}

void TimePointSampler::checkLevel(Index level) const
{
  if (level < 0 || level > numSteps())
  {
    throw std::runtime_error("TimePointSampler time level is out of range");
  }
}

void TimePointSampler::checkInputs(
    const Vector<Real>& state,
    const Vector<Real>& prm) const
{
  if (state.size() != numStates() || prm.size() != numParams())
  {
    throw std::runtime_error("TimePointSampler input size mismatch");
  }
}

std::vector<TimePointSampler::Stencil> TimePointSampler::buildStencils(
    const MixedFieldView&      field,
    const std::vector<Point3>& points,
    const Vector<Index>&       components)
{
  std::vector<Stencil> stencils;
  stencils.reserve(points.size() * static_cast<std::size_t>(components.size()));

  for (const Point3& point : points)
  {
    const ScalarStencil scalar = findScalarStencil(field.space(), point);
    for (Index component : components)
    {
      Stencil stencil;
      stencil.indices.reserve(scalar.weights.size());
      stencil.weights.reserve(scalar.weights.size());
      for (Index i = 0; i < scalar.weights.size(); ++i)
      {
        stencil.indices.push_back(
            field.globalDof(scalar.node_ids[static_cast<std::size_t>(i)],
                            component));
        stencil.weights.push_back(scalar.weights[i]);
      }
      stencils.push_back(std::move(stencil));
    }
  }

  return stencils;
}

void TimePointSampler::resize(Vector<Real>& out,
                              Index         size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

} // namespace inverse
} // namespace femx
