#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/fem/Cell.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/linalg/MatrixView.hpp>
#include <femx/linalg/VectorView.hpp>

using namespace std;

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
  Vector<Index> nids;
  Vector<Real>  wts;
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
      lower = min(lower, cell.node(in)[a]);
      upper = max(upper, cell.node(in)[a]);
    }
    if (point[a] < lower - point_tol || point[a] > upper + point_tol)
    {
      return false;
    }
  }
  return true;
}

bool insideSimplex(const Vector<Real>& wts)
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
  return abs(sum - 1.0) <= 10.0 * point_tol;
}

Vector<Real> shapeWeights(const FiniteElement&   fe,
                          const QuadraturePoint& qp)
{
  Vector<Real> wts(fe.numDofsPerElement());
  fe.calcShape(qp, VectorView<Real>(wts.data(), wts.size()));
  return wts;
}

bool triWeights(const FiniteElement& fe,
                const Cell&          cell,
                const Point3&        point,
                Vector<Real>&        wts)
{
  const Point3 a   = cell.node(0);
  const Point3 e1  = difference(cell.node(1), a);
  const Point3 e2  = difference(cell.node(2), a);
  const Point3 rhs = difference(point, a);

  const Real det = e1[0] * e2[1] - e1[1] * e2[0];
  if (abs(det) < det_tol)
  {
    return false;
  }

  const Real r = (rhs[0] * e2[1] - rhs[1] * e2[0]) / det;
  const Real s = (e1[0] * rhs[1] - e1[1] * rhs[0]) / det;

  wts = shapeWeights(fe, QuadraturePoint{{r, s, 0.0}, 0.0});
  return insideSimplex(wts);
}

bool tetWeights(const FiniteElement& fe,
                const Cell&          cell,
                const Point3&        point,
                Vector<Real>&        wts)
{
  const Point3 a   = cell.node(0);
  const Point3 e1  = difference(cell.node(1), a);
  const Point3 e2  = difference(cell.node(2), a);
  const Point3 e3  = difference(cell.node(3), a);
  const Point3 rhs = difference(point, a);

  const Real det = dot(e1, cross(e2, e3));
  if (abs(det) < det_tol)
  {
    return false;
  }

  const Real r = dot(rhs, cross(e2, e3)) / det;
  const Real s = dot(e1, cross(rhs, e3)) / det;
  const Real t = dot(e1, cross(e2, rhs)) / det;

  wts = shapeWeights(fe, QuadraturePoint{{r, s, t}, 0.0});
  return insideSimplex(wts);
}

Point3 mappedPoint(const Cell&         cell,
                   const Vector<Real>& wts,
                   Index               dim)
{
  Point3 mapped{0.0, 0.0, 0.0};
  for (Index in = 0; in < cell.numNodes(); ++in)
  {
    for (Index a = 0; a < dim; ++a)
    {
      mapped[a] += wts[in] * cell.node(in)[a];
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
                   Vector<Real>&        wts)
{
  const QuadraturePoint qp{{r, s, 0.0}, 0.0};
  wts = shapeWeights(fe, qp);

  Vector<Real> grad(fe.numDofsPerElement() * fe.dim());
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
    const Real x  = cell.node(in)[0];
    const Real y  = cell.node(in)[1];
    j00          += x * grad[in * fe.dim()];
    j01          += x * grad[in * fe.dim() + 1];
    j10          += y * grad[in * fe.dim()];
    j11          += y * grad[in * fe.dim() + 1];
  }

  const Point3 phys = mappedPoint(cell, wts, fe.dim());
  const Real   res0 = phys[0] - point[0];
  const Real   res1 = phys[1] - point[1];
  const Real   det  = j00 * j11 - j01 * j10;
  if (abs(det) < det_tol)
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
                 Vector<Real>&        wts)
{
  Real r = 0.0;
  Real s = 0.0;

  for (Index iter = 0; iter < 12; ++iter)
  {
    Real dr = 0.0;
    Real ds = 0.0;
    if (!quadSolveStep(cell, fe, r, s, point, dr, ds, wts))
    {
      return false;
    }

    r -= dr;
    s -= ds;

    if (abs(dr) + abs(ds) <= point_tol)
    {
      break;
    }
  }

  wts               = shapeWeights(fe, QuadraturePoint{{r, s, 0.0}, 0.0});
  const Point3 phys = mappedPoint(cell, wts, fe.dim());
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
                 Vector<Real>&        wts)
{
  switch (fe.referenceElement())
  {
  case ReferenceElement::Triangle:
    return triWeights(fe, cell, point, wts);

  case ReferenceElement::Quadrilateral:
    return quadWeights(fe, cell, point, wts);

  case ReferenceElement::Tetrahedron:
    return tetWeights(fe, cell, point, wts);

  case ReferenceElement::Segment:
    break;
  }

  throw runtime_error(
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
    throw runtime_error(
        "TimePointInterpolator mesh dimension does not match finite element");
  }

  for (Index ic = 0; ic < mesh.numElems(); ++ic)
  {
    const Cell& cell = mesh.cell(ic);
    if (cell.numNodes() != fe.numNodes())
    {
      throw runtime_error(
          "TimePointInterpolator cell node count does not match finite element");
    }
    if (!insideBox(cell, point, mesh.dim()))
    {
      continue;
    }

    Vector<Real> wts;
    if (cellWeights(fe, cell, point, wts))
    {
      out = ScalarStencil{cell.nodeIds(), wts};
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
  throw runtime_error("TimePointInterpolator point is outside the mesh");
}

} // namespace

TimePointInterpolator::TimePointInterpolator(Index               nt,
                                             const MixedFESpace& space,
                                             Index               fid,
                                             Vector<Point3>      pts,
                                             Vector<Index>       comps,
                                             Index               nprm)
  : nt_(nt),
    nst_(space.numDofs()),
    nprm_(nprm),
    pts_(std::move(pts)),
    comps_(std::move(comps))
{
  if (nt_ < 0 || nst_ < 0 || nprm_ < 0)
  {
    throw runtime_error("TimePointInterpolator received invalid dimensions");
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
      throw runtime_error(
          "TimePointInterpolator component is out of range");
    }
  }

  stencils_ = buildStencils(field, pts_, comps_);
}

Index TimePointInterpolator::numSteps() const
{
  return nt_;
}

Index TimePointInterpolator::numStates() const
{
  return nst_;
}

Index TimePointInterpolator::numParams() const
{
  return nprm_;
}

Index TimePointInterpolator::numObservations() const
{
  return stencils_.size();
}

void TimePointInterpolator::observe(Index               level,
                                    const Vector<Real>& state,
                                    const Vector<Real>& prm,
                                    Vector<Real>&       out) const
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

void TimePointInterpolator::applyStateJac(Index               level,
                                          const Vector<Real>& state,
                                          const Vector<Real>& prm,
                                          const Vector<Real>& dir,
                                          Vector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numStates())
  {
    throw runtime_error(
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

void TimePointInterpolator::applyStateJacT(Index               level,
                                           const Vector<Real>& state,
                                           const Vector<Real>& prm,
                                           const Vector<Real>& dir,
                                           Vector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numObservations())
  {
    throw runtime_error(
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

void TimePointInterpolator::applyParamJac(Index               level,
                                          const Vector<Real>& state,
                                          const Vector<Real>& prm,
                                          const Vector<Real>& dir,
                                          Vector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numParams())
  {
    throw runtime_error(
        "TimePointInterpolator parameter direction size mismatch");
  }

  resizeOrZero(out, numObservations());
}

void TimePointInterpolator::applyParamJacT(Index               level,
                                           const Vector<Real>& state,
                                           const Vector<Real>& prm,
                                           const Vector<Real>& dir,
                                           Vector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (dir.size() != numObservations())
  {
    throw runtime_error(
        "TimePointInterpolator observation direction size mismatch");
  }

  resizeOrZero(out, numParams());
}

const Vector<Point3>& TimePointInterpolator::pts() const
{
  return pts_;
}

const Vector<Index>& TimePointInterpolator::comps() const
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

Vector<Point3> TimePointInterpolator::filterPointsInside(
    const MixedFESpace&   space,
    Index                 fid,
    const Vector<Point3>& pts)
{
  Vector<Point3> filtered;
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
    throw runtime_error("TimePointInterpolator time level is out of range");
  }
}

void TimePointInterpolator::checkInputs(
    const Vector<Real>& state,
    const Vector<Real>& prm) const
{
  if (state.size() != numStates() || prm.size() != numParams())
  {
    throw runtime_error("TimePointInterpolator input size mismatch");
  }
}

Vector<TimePointInterpolator::Stencil> TimePointInterpolator::buildStencils(
    const MixedFieldView& field,
    const Vector<Point3>& pts,
    const Vector<Index>&  comps)
{
  Vector<Stencil> stencils;
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
