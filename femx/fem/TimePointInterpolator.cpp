#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/fem/Element.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/linalg/MatrixView.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{
namespace fem
{

namespace
{

constexpr Real point_tol = 1.0e-10;
constexpr Real det_tol   = 1.0e-14;

struct ScalarStencil
{
  Array<Index> nids;
  HostVector   wts;
};

bool insideBox(const Element& elem,
               const Point3&  point,
               Index          dim)
{
  for (Index a = 0; a < dim; ++a)
  {
    Real lower = elem.node(0)[a];
    Real upper = elem.node(0)[a];
    for (Index in = 1; in < elem.numNodes(); ++in)
    {
      lower = std::min(lower, elem.node(in)[a]);
      upper = std::max(upper, elem.node(in)[a]);
    }
    if (point[a] < lower - point_tol || point[a] > upper + point_tol)
    {
      return false;
    }
  }
  return true;
}

bool insideSimplex(const HostVector& wts)
{
  Real sum = 0.0;
  for (Real wt : wts)
  {
    if (wt < -point_tol || wt > 1.0 + point_tol)
    {
      return false;
    }
    sum += wt;
  }
  return std::abs(sum - 1.0) <= 10.0 * point_tol;
}

HostVector shapeWeights(const FiniteElement&   fe,
                        const QuadraturePoint& qp)
{
  HostVector wts(fe.numDofsPerElement());
  fe.calcN(qp, HostVectorView(wts.data(), wts.size()));
  return wts;
}

bool triWeights(const FiniteElement& fe,
                const Element&       elem,
                const Point3&        point,
                HostVector&          wts)
{
  const Point3 a   = elem.node(0);
  const Point3 e1  = difference(elem.node(1), a);
  const Point3 e2  = difference(elem.node(2), a);
  const Point3 rhs = difference(point, a);

  const Real det = e1[0] * e2[1] - e1[1] * e2[0];
  if (std::abs(det) < det_tol)
  {
    return false;
  }

  const Real r = (rhs[0] * e2[1] - rhs[1] * e2[0]) / det;
  const Real s = (e1[0] * rhs[1] - e1[1] * rhs[0]) / det;

  wts = shapeWeights(fe, QuadraturePoint{{r, s, 0.0}, 0.0});
  return insideSimplex(wts);
}

bool tetWeights(const FiniteElement& fe,
                const Element&       elem,
                const Point3&        point,
                HostVector&          wts)
{
  const Point3 a   = elem.node(0);
  const Point3 e1  = difference(elem.node(1), a);
  const Point3 e2  = difference(elem.node(2), a);
  const Point3 e3  = difference(elem.node(3), a);
  const Point3 rhs = difference(point, a);

  const Real det = dot(e1, cross(e2, e3));
  if (std::abs(det) < det_tol)
  {
    return false;
  }

  const Real r = dot(rhs, cross(e2, e3)) / det;
  const Real s = dot(e1, cross(rhs, e3)) / det;
  const Real t = dot(e1, cross(e2, rhs)) / det;

  wts = shapeWeights(fe, QuadraturePoint{{r, s, t}, 0.0});
  return insideSimplex(wts);
}

Point3 mappedPoint(const Element&    elem,
                   const HostVector& wts,
                   Index             dim)
{
  Point3 mapped{0.0, 0.0, 0.0};
  for (Index in = 0; in < elem.numNodes(); ++in)
  {
    for (Index a = 0; a < dim; ++a)
    {
      mapped[a] += wts[in] * elem.node(in)[a];
    }
  }
  return mapped;
}

bool quadSolveStep(const Element&       elem,
                   const FiniteElement& fe,
                   Real                 r,
                   Real                 s,
                   const Point3&        point,
                   Real&                dr,
                   Real&                ds,
                   HostVector&          wts)
{
  const QuadraturePoint qp{{r, s, 0.0}, 0.0};
  wts = shapeWeights(fe, qp);

  HostVector grad(fe.numDofsPerElement() * fe.dim());
  fe.calcdNdr(
      qp,
      MatrixView<Real>(
          grad.data(), fe.numDofsPerElement(), fe.dim()));

  Real j00 = 0.0;
  Real j01 = 0.0;
  Real j10 = 0.0;
  Real j11 = 0.0;
  for (Index in = 0; in < elem.numNodes(); ++in)
  {
    const Real x  = elem.node(in)[0];
    const Real y  = elem.node(in)[1];
    j00          += x * grad[in * fe.dim()];
    j01          += x * grad[in * fe.dim() + 1];
    j10          += y * grad[in * fe.dim()];
    j11          += y * grad[in * fe.dim() + 1];
  }

  const Point3 phys = mappedPoint(elem, wts, fe.dim());
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
                 const Element&       elem,
                 const Point3&        point,
                 HostVector&          wts)
{
  Real r = 0.0;
  Real s = 0.0;

  for (Index iter = 0; iter < 12; ++iter)
  {
    Real dr = 0.0;
    Real ds = 0.0;
    if (!quadSolveStep(elem, fe, r, s, point, dr, ds, wts))
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

  wts               = shapeWeights(fe, QuadraturePoint{{r, s, 0.0}, 0.0});
  const Point3 phys = mappedPoint(elem, wts, fe.dim());
  const Real   err0 = phys[0] - point[0];
  const Real   err1 = phys[1] - point[1];
  const bool   inside =
      r >= -1.0 - point_tol && r <= 1.0 + point_tol
      && s >= -1.0 - point_tol && s <= 1.0 + point_tol;
  return inside
         && (err0 * err0 + err1 * err1 <= 100.0 * point_tol * point_tol);
}

bool elemWeights(const FiniteElement& fe,
                 const Element&       elem,
                 const Point3&        point,
                 HostVector&          wts)
{
  switch (fe.referenceElement())
  {
  case ReferenceElement::Triangle:
    return triWeights(fe, elem, point, wts);

  case ReferenceElement::Quadrilateral:
    return quadWeights(fe, elem, point, wts);

  case ReferenceElement::Tetrahedron:
    return tetWeights(fe, elem, point, wts);

  case ReferenceElement::Segment:
    break;
  }

  throw std::runtime_error(
      "TimePointInterpolator does not support this reference element");
}

bool tryFindScalarStencil(const FESpace& space,
                          const Point3&  point,
                          ScalarStencil& out)
{
  const Mesh&          mesh = space.mesh();
  const FiniteElement& fe   = space.finiteElement();
  if (mesh.dim() != fe.dim())
  {
    throw std::runtime_error(
        "TimePointInterpolator mesh dimension does not match finite element");
  }

  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    const Element& elem = mesh.elem(ie);
    if (elem.numNodes() != fe.numNodes())
    {
      throw std::runtime_error(
          "TimePointInterpolator elem node count does not match finite element");
    }
    if (!insideBox(elem, point, mesh.dim()))
    {
      continue;
    }

    HostVector wts;
    if (elemWeights(fe, elem, point, wts))
    {
      out = ScalarStencil{elem.nodeIds(), wts};
      return true;
    }
  }

  return false;
}

ScalarStencil findScalarStencil(const FESpace& space,
                                const Point3&  point)
{
  ScalarStencil stencil;
  if (tryFindScalarStencil(space, point, stencil))
  {
    return stencil;
  }
  throw std::runtime_error("TimePointInterpolator point is outside the mesh");
}

} // namespace

TimePointInterpolator::TimePointInterpolator(Index               num_steps,
                                             const MixedFESpace& space,
                                             Index               fid,
                                             Array<Point3>       pts,
                                             Array<Index>        comps,
                                             Index               num_param)
  : num_steps_(num_steps),
    num_states_(space.numDofs()),
    num_param_(num_param),
    pts_(std::move(pts)),
    comps_(std::move(comps))
{
  if (num_steps_ < 0 || num_states_ < 0 || num_param_ < 0)
  {
    throw std::runtime_error("TimePointInterpolator received invalid dimensions");
  }

  const MixedFieldView field = space.field(fid);
  if (comps_.empty())
  {
    for (Index c = 0; c < field.numComponents(); ++c)
    {
      comps_.push_back(c);
    }
  }

  for (Index comp : comps_)
  {
    if (comp < 0 || comp >= field.numComponents())
    {
      throw std::runtime_error(
          "TimePointInterpolator component is out of range");
    }
  }

  stencils_ = buildStencils(field, pts_, comps_);
}

Index TimePointInterpolator::numSteps() const
{
  return num_steps_;
}

Index TimePointInterpolator::numStates() const
{
  return num_states_;
}

Index TimePointInterpolator::numParams() const
{
  return num_param_;
}

Index TimePointInterpolator::numObservations() const
{
  return stencils_.size();
}

void TimePointInterpolator::observe(Index             level,
                                    const HostVector& state,
                                    const HostVector& prm,
                                    HostVector&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  resizeOrZero(out, numObservations());

  for (Index i = 0; i < numObservations(); ++i)
  {
    const Stencil& stencil = stencils_[i];
    for (Index j = 0; j < stencil.indices.size(); ++j)
    {
      out[i] += stencil.wts[j] * state[stencil.indices[j]];
    }
  }
}

void TimePointInterpolator::applyStateJac(Index             level,
                                          const HostVector& state,
                                          const HostVector& prm,
                                          const HostVector& dir,
                                          HostVector&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numStates())
  {
    throw std::runtime_error(
        "TimePointInterpolator state direction size mismatch");
  }

  resizeOrZero(out, numObservations());
  for (Index i = 0; i < numObservations(); ++i)
  {
    const Stencil& stencil = stencils_[i];
    for (Index j = 0; j < stencil.indices.size(); ++j)
    {
      out[i] += stencil.wts[j] * dir[stencil.indices[j]];
    }
  }
}

void TimePointInterpolator::applyStateJacT(Index             level,
                                           const HostVector& state,
                                           const HostVector& prm,
                                           const HostVector& dir,
                                           HostVector&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numObservations())
  {
    throw std::runtime_error(
        "TimePointInterpolator observation direction size mismatch");
  }

  resizeOrZero(out, numStates());
  for (Index i = 0; i < numObservations(); ++i)
  {
    const Stencil& stencil = stencils_[i];
    for (Index j = 0; j < stencil.indices.size(); ++j)
    {
      out[stencil.indices[j]] += stencil.wts[j] * dir[i];
    }
  }
}

void TimePointInterpolator::applyParamJac(Index             level,
                                          const HostVector& state,
                                          const HostVector& prm,
                                          const HostVector& dir,
                                          HostVector&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numParams())
  {
    throw std::runtime_error(
        "TimePointInterpolator parameter direction size mismatch");
  }

  resizeOrZero(out, numObservations());
}

void TimePointInterpolator::applyParamJacT(Index             level,
                                           const HostVector& state,
                                           const HostVector& prm,
                                           const HostVector& dir,
                                           HostVector&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numObservations())
  {
    throw std::runtime_error(
        "TimePointInterpolator observation direction size mismatch");
  }

  resizeOrZero(out, numParams());
}

const Array<Point3>& TimePointInterpolator::pts() const
{
  return pts_;
}

const Array<Index>& TimePointInterpolator::comps() const
{
  return comps_;
}

bool TimePointInterpolator::containsPoint(const MixedFESpace& space,
                                          Index               fid,
                                          const Point3&       point)
{
  const MixedFieldView field = space.field(fid);
  ScalarStencil        stencil;
  return tryFindScalarStencil(field.space(), point, stencil);
}

Array<Point3> TimePointInterpolator::filterPointsInside(
    const MixedFESpace&  space,
    Index                fid,
    const Array<Point3>& pts)
{
  Array<Point3> filtered;
  filtered.reserve(pts.size());
  for (const Point3& point : pts)
  {
    if (containsPoint(space, fid, point))
    {
      filtered.push_back(point);
    }
  }
  return filtered;
}

void TimePointInterpolator::checkLevel(Index level) const
{
  if (level < 0 || level > numSteps())
  {
    throw std::runtime_error("TimePointInterpolator time level is out of range");
  }
}

void TimePointInterpolator::checkInputs(
    const HostVector& state,
    const HostVector& prm) const
{
  if (state.size() != numStates() || prm.size() != numParams())
  {
    throw std::runtime_error("TimePointInterpolator input size mismatch");
  }
}

Array<TimePointInterpolator::Stencil> TimePointInterpolator::buildStencils(
    const MixedFieldView& field,
    const Array<Point3>&  pts,
    const Array<Index>&   comps)
{
  Array<Stencil> stencils;
  stencils.reserve(pts.size() * comps.size());

  for (const Point3& point : pts)
  {
    const ScalarStencil scalar = findScalarStencil(field.space(), point);
    for (Index comp : comps)
    {
      Stencil stencil;
      stencil.indices.reserve(scalar.wts.size());
      stencil.wts.reserve(scalar.wts.size());
      for (Index i = 0; i < scalar.wts.size(); ++i)
      {
        stencil.indices.push_back(
            field.globalDof(scalar.nids[i], comp));
        stencil.wts.push_back(scalar.wts[i]);
      }
      stencils.push_back(std::move(stencil));
    }
  }

  return stencils;
}

} // namespace fem
} // namespace femx
