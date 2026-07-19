#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/fem/Element.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/linalg/View.hpp>

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
      HostMatrixView<Real>(
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
  require(mesh.dim() == fe.dim(),
          "TimePointInterpolator mesh dimension does not match finite element");

  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    const Element& elem = mesh.elem(ie);
    require(elem.numNodes() == fe.numNodes(),
            "TimePointInterpolator elem node count does not match finite element");
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

Index DeviceTimePointInterpolator::numSteps() const
{
  return num_steps_;
}

Index DeviceTimePointInterpolator::numStates() const
{
  return data_.numStates();
}

Index DeviceTimePointInterpolator::numObservations() const
{
  return data_.numObservations();
}

void DeviceTimePointInterpolator::observe(Index                 level,
                                          DeviceConstVectorView state,
                                          DeviceVectorView      out,
                                          CudaContext&          ctx) const
{
  checkLevel(level);
  femx::apply(data_.matrix(), state, out, ctx);
}

void DeviceTimePointInterpolator::addStateJacT(
    Index                 level,
    DeviceConstVectorView dir,
    DeviceVectorView      out,
    CudaContext&          ctx) const
{
  checkLevel(level);
  femx::applyT(data_.matrix(), dir, out, ctx, 1.0, 1.0);
}

void DeviceTimePointInterpolator::checkLevel(Index level) const
{
  require(level >= 0 && level <= numSteps(),
          "DeviceTimePointInterpolator time level is out of range");
}

TimePointInterpolator::TimePointInterpolator(Index               num_steps,
                                             const MixedFESpace& space,
                                             Index               fid,
                                             Array<Point3>       pts,
                                             Array<Index>        comps,
                                             Index               num_prm)
  : num_steps_(num_steps),
    num_prm_(num_prm),
    pts_(std::move(pts)),
    comps_(std::move(comps))
{
  require(num_steps_ >= 0 && space.numDofs() >= 0 && num_prm_ >= 0,
          "TimePointInterpolator received invalid dimensions");

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
    require(comp >= 0 && comp < field.numComponents(),
            "TimePointInterpolator component is out of range");
  }

  data_ = buildData(field, space.numDofs(), pts_, comps_);
}

Index TimePointInterpolator::numSteps() const
{
  return num_steps_;
}

Index TimePointInterpolator::numStates() const
{
  return data_.numStates();
}

Index TimePointInterpolator::numParams() const
{
  return num_prm_;
}

Index TimePointInterpolator::numObservations() const
{
  return data_.numObservations();
}

std::unique_ptr<DeviceTimeObservationOperator>
TimePointInterpolator::copyToDevice(CudaContext& ctx) const
{
  auto out = std::make_unique<DeviceTimePointInterpolator>();
  copy(*this, *out, ctx);
  return out;
}

void TimePointInterpolator::observe(Index             level,
                                    const HostVector& state,
                                    const HostVector& prm,
                                    HostVector&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (out.size() != numObservations())
  {
    out.resize(numObservations());
  }
  CpuContext ctx;
  femx::apply(data_.matrix(), state.view(), out.view(), ctx);
}

void TimePointInterpolator::applyStateJac(Index             level,
                                          const HostVector& state,
                                          const HostVector& prm,
                                          const HostVector& dir,
                                          HostVector&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  require(dir.size() == numStates(),
          "TimePointInterpolator state direction size mismatch");

  if (out.size() != numObservations())
  {
    out.resize(numObservations());
  }
  CpuContext ctx;
  femx::apply(data_.matrix(), dir.view(), out.view(), ctx);
}

void TimePointInterpolator::applyStateJacT(Index             level,
                                           const HostVector& state,
                                           const HostVector& prm,
                                           const HostVector& dir,
                                           HostVector&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  require(dir.size() == numObservations(),
          "TimePointInterpolator observation direction size mismatch");

  resizeOrZero(out, numStates());
  CpuContext ctx;
  femx::applyT(data_.matrix(), dir.view(), out.view(), ctx, 1.0, 1.0);
}

void TimePointInterpolator::applyParamJac(Index             level,
                                          const HostVector& state,
                                          const HostVector& prm,
                                          const HostVector& dir,
                                          HostVector&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  require(dir.size() == numParams(),
          "TimePointInterpolator parameter direction size mismatch");

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
  require(dir.size() == numObservations(),
          "TimePointInterpolator observation direction size mismatch");

  resizeOrZero(out, numParams());
}

const HostPointInterpolatorData& TimePointInterpolator::data() const noexcept
{
  return data_;
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
  require(level >= 0 && level <= numSteps(),
          "TimePointInterpolator time level is out of range");
}

void TimePointInterpolator::checkInputs(
    const HostVector& state,
    const HostVector& prm) const
{
  require(state.size() == numStates() && prm.size() == numParams(),
          "TimePointInterpolator input size mismatch");
}

HostPointInterpolatorData TimePointInterpolator::buildData(
    const MixedFieldView& field,
    Index                 num_states,
    const Array<Point3>&  pts,
    const Array<Index>&   comps)
{
  const Index     num_obs = pts.size() * comps.size();
  HostIndexVector offsets;
  HostIndexVector dofs;
  HostVector      wts;
  offsets.reserve(num_obs + 1);
  dofs.reserve(num_obs * field.numShapesPerElem());
  wts.reserve(num_obs * field.numShapesPerElem());
  offsets.push_back(0);

  for (const Point3& point : pts)
  {
    const ScalarStencil scalar = findScalarStencil(field.space(), point);
    for (Index comp : comps)
    {
      for (Index i = 0; i < scalar.wts.size(); ++i)
      {
        dofs.push_back(field.globalDof(scalar.nids[i], comp));
        wts.push_back(scalar.wts[i]);
      }
      offsets.push_back(dofs.size());
    }
  }

  HostCsrGraph  graph(num_obs,
                     num_states,
                     std::move(offsets),
                     std::move(dofs));
  HostCsrMatrix mat(graph);
  mat.vals() = std::move(wts);
  HostPointInterpolatorData data;
  data.mat_ = std::move(mat);
  return data;
}

} // namespace fem
} // namespace femx
