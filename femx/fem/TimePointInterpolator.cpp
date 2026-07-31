#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/common/View.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/native/HostSystemMatrix.hpp>

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
  HostVector<Index> nids;
  HostVector<Real>  wts;
};

bool insideBox(const Mesh&   mesh,
               Index         ie,
               const Point3& point)
{
  for (Index id = 0; id < mesh.dim(); ++id)
  {
    Real lower = mesh.elemNode(ie, 0)[id];
    Real upper = mesh.elemNode(ie, 0)[id];
    for (Index in = 1; in < mesh.elemNumNodes(ie); ++in)
    {
      lower = std::min(lower, mesh.elemNode(ie, in)[id]);
      upper = std::max(upper, mesh.elemNode(ie, in)[id]);
    }
    if (point[id] < lower - point_tol || point[id] > upper + point_tol)
    {
      return false;
    }
  }
  return true;
}

bool insideSimplex(const HostVector<Real>& wts)
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

HostVector<Real> shapeWeights(const FiniteElement&   fe,
                              const QuadraturePoint& qp)
{
  HostVector<Real> wts(fe.numDofsPerElement());
  fe.calcN(qp, HostVectorView<Real>(wts.data(), wts.size()));
  return wts;
}

bool triWeights(const FiniteElement& fe,
                const Mesh&          mesh,
                Index                ie,
                const Point3&        point,
                HostVector<Real>&    wts)
{
  const Point3 a   = mesh.elemNode(ie, 0);
  const Point3 e1  = difference(mesh.elemNode(ie, 1), a);
  const Point3 e2  = difference(mesh.elemNode(ie, 2), a);
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
                const Mesh&          mesh,
                Index                ie,
                const Point3&        point,
                HostVector<Real>&    wts)
{
  const Point3 a   = mesh.elemNode(ie, 0);
  const Point3 e1  = difference(mesh.elemNode(ie, 1), a);
  const Point3 e2  = difference(mesh.elemNode(ie, 2), a);
  const Point3 e3  = difference(mesh.elemNode(ie, 3), a);
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

Point3 mappedPoint(const Mesh&             mesh,
                   Index                   ie,
                   const HostVector<Real>& wts,
                   Index                   dim)
{
  Point3 mapped{0.0, 0.0, 0.0};
  for (Index in = 0; in < mesh.elemNumNodes(ie); ++in)
  {
    for (Index id = 0; id < dim; ++id)
    {
      mapped[id] += wts[in] * mesh.elemNode(ie, in)[id];
    }
  }
  return mapped;
}

bool quadSolveStep(const Mesh&          mesh,
                   Index                ie,
                   const FiniteElement& fe,
                   Real                 r,
                   Real                 s,
                   const Point3&        point,
                   Real&                dr,
                   Real&                ds,
                   HostVector<Real>&    wts)
{
  const QuadraturePoint qp{{r, s, 0.0}, 0.0};
  wts = shapeWeights(fe, qp);

  HostVector<Real> grad(fe.numDofsPerElement() * fe.dim());
  fe.calcdNdr(
      qp,
      HostMatrixView<Real>(
          grad.data(), fe.numDofsPerElement(), fe.dim()));

  Real j00 = 0.0;
  Real j01 = 0.0;
  Real j10 = 0.0;
  Real j11 = 0.0;
  for (Index in = 0; in < mesh.elemNumNodes(ie); ++in)
  {
    const Real x  = mesh.elemNode(ie, in)[0];
    const Real y  = mesh.elemNode(ie, in)[1];
    j00          += x * grad[in * fe.dim()];
    j01          += x * grad[in * fe.dim() + 1];
    j10          += y * grad[in * fe.dim()];
    j11          += y * grad[in * fe.dim() + 1];
  }

  const Point3 phys = mappedPoint(mesh, ie, wts, fe.dim());
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
                 const Mesh&          mesh,
                 Index                ie,
                 const Point3&        point,
                 HostVector<Real>&    wts)
{
  Real r = 0.0;
  Real s = 0.0;

  for (Index iter = 0; iter < 12; ++iter)
  {
    Real dr = 0.0;
    Real ds = 0.0;
    if (!quadSolveStep(mesh, ie, fe, r, s, point, dr, ds, wts))
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
  const Point3 phys = mappedPoint(mesh, ie, wts, fe.dim());
  const Real   err0 = phys[0] - point[0];
  const Real   err1 = phys[1] - point[1];
  const bool   inside =
      r >= -1.0 - point_tol && r <= 1.0 + point_tol
      && s >= -1.0 - point_tol && s <= 1.0 + point_tol;
  return inside
         && (err0 * err0 + err1 * err1 <= 100.0 * point_tol * point_tol);
}

bool elemWeights(const FiniteElement& fe,
                 const Mesh&          mesh,
                 Index                ie,
                 const Point3&        point,
                 HostVector<Real>&    wts)
{
  switch (fe.shape())
  {
  case ElementShape::Triangle:
    return triWeights(fe, mesh, ie, point, wts);

  case ElementShape::Quadrilateral:
    return quadWeights(fe, mesh, ie, point, wts);

  case ElementShape::Tetrahedron:
    return tetWeights(fe, mesh, ie, point, wts);

  case ElementShape::Unknown:
  case ElementShape::Segment:
  case ElementShape::Hexahedron:
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
    require(mesh.elemNumNodes(ie) == fe.numNodes(),
            "TimePointInterpolator elem node count does not match finite element");
    if (!insideBox(mesh, ie, point))
    {
      continue;
    }

    HostVector<Real> wts;
    if (elemWeights(fe, mesh, ie, point, wts))
    {
      out = ScalarStencil{HostVector<Index>(mesh.elemNodeIds(ie)), wts};
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

void DeviceTimePointInterpolator::observe(Index                        level,
                                          DeviceVectorView<const Real> state,
                                          DeviceVectorView<Real>       out,
                                          linalg::CudaContext&         ctx) const
{
  checkLevel(level);
  linalg::CudaSystemMatrix jac(ctx);
  jac.apply(data_.matrix(), state, out);
}

void DeviceTimePointInterpolator::addStateJacT(
    Index                        level,
    DeviceVectorView<const Real> dir,
    DeviceVectorView<Real>       out,
    linalg::CudaContext&         ctx) const
{
  checkLevel(level);
  linalg::CudaSystemMatrix jac(ctx);
  jac.applyT(data_.matrix(), dir, out, 1.0, 1.0);
}

void DeviceTimePointInterpolator::checkLevel(Index level) const
{
  require(level >= 0 && level <= numSteps(),
          "DeviceTimePointInterpolator time level is out of range");
}

TimePointInterpolator::TimePointInterpolator(Index               num_steps,
                                             const MixedFESpace& space,
                                             Index               fid,
                                             HostVector<Point3>  pts,
                                             HostVector<Index>   comps,
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
    for (Index ic = 0; ic < field.numComponents(); ++ic)
    {
      comps_.push_back(ic);
    }
  }

  for (Index ic : comps_)
  {
    require(ic >= 0 && ic < field.numComponents(),
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
TimePointInterpolator::copyToDevice(linalg::CudaContext& ctx) const
{
  auto out = std::make_unique<DeviceTimePointInterpolator>();
  copy(*this, *out, ctx);
  return out;
}

void TimePointInterpolator::observe(Index                   level,
                                    const HostVector<Real>& state,
                                    const HostVector<Real>& prm,
                                    HostVector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  if (out.size() != numObservations())
  {
    out.resize(numObservations());
  }
  linalg::HostContext      ctx;
  linalg::HostSystemMatrix jac(ctx);
  jac.apply(data_.matrix(), state.view(), out.view());
}

void TimePointInterpolator::applyStateJac(Index                   level,
                                          const HostVector<Real>& state,
                                          const HostVector<Real>& prm,
                                          const HostVector<Real>& dir,
                                          HostVector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  require(dir.size() == numStates(),
          "TimePointInterpolator state direction size mismatch");

  if (out.size() != numObservations())
  {
    out.resize(numObservations());
  }
  linalg::HostContext      ctx;
  linalg::HostSystemMatrix jac(ctx);
  jac.apply(data_.matrix(), dir.view(), out.view());
}

void TimePointInterpolator::applyStateJacT(Index                   level,
                                           const HostVector<Real>& state,
                                           const HostVector<Real>& prm,
                                           const HostVector<Real>& dir,
                                           HostVector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  require(dir.size() == numObservations(),
          "TimePointInterpolator observation direction size mismatch");

  linalg::HostContext      ctx;
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::HostSystemMatrix jac(ctx);
  vec_handler.assign(out, numStates(), 0);
  jac.applyT(data_.matrix(), dir.view(), out.view(), 1.0, 1.0);
}

void TimePointInterpolator::applyParamJac(Index                   level,
                                          const HostVector<Real>& state,
                                          const HostVector<Real>& prm,
                                          const HostVector<Real>& dir,
                                          HostVector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  require(dir.size() == numParams(),
          "TimePointInterpolator parameter direction size mismatch");

  linalg::HostContext ctx;
  auto&               vec_handler = ctx.vectorHandler();
  vec_handler.assign(out, numObservations(), 0);
}

void TimePointInterpolator::applyParamJacT(Index                   level,
                                           const HostVector<Real>& state,
                                           const HostVector<Real>& prm,
                                           const HostVector<Real>& dir,
                                           HostVector<Real>&       out) const
{
  checkLevel(level);
  checkInputs(state, prm);
  require(dir.size() == numObservations(),
          "TimePointInterpolator observation direction size mismatch");

  linalg::HostContext ctx;
  auto&               vec_handler = ctx.vectorHandler();
  vec_handler.assign(out, numParams(), 0);
}

const HostPointInterpolatorData& TimePointInterpolator::data() const noexcept
{
  return data_;
}

const HostVector<Point3>& TimePointInterpolator::pts() const
{
  return pts_;
}

const HostVector<Index>& TimePointInterpolator::comps() const
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

HostVector<Point3> TimePointInterpolator::filterPointsInside(
    const MixedFESpace&       space,
    Index                     fid,
    const HostVector<Point3>& pts)
{
  HostVector<Point3> filtered;
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
    const HostVector<Real>& state,
    const HostVector<Real>& prm) const
{
  require(state.size() == numStates() && prm.size() == numParams(),
          "TimePointInterpolator input size mismatch");
}

HostPointInterpolatorData TimePointInterpolator::buildData(
    const MixedFieldView&     field,
    Index                     num_states,
    const HostVector<Point3>& pts,
    const HostVector<Index>&  comps)
{
  const Index       num_obs = pts.size() * comps.size();
  HostVector<Index> offsets;
  HostVector<Index> dofs;
  HostVector<Real>  wts;
  offsets.reserve(num_obs + 1);
  dofs.reserve(num_obs * field.numShapesPerElem());
  wts.reserve(num_obs * field.numShapesPerElem());
  offsets.push_back(0);

  for (const Point3& point : pts)
  {
    const ScalarStencil scalar = findScalarStencil(field.space(), point);
    for (Index ic : comps)
    {
      for (Index in = 0; in < scalar.wts.size(); ++in)
      {
        dofs.push_back(field.globalDof(scalar.nids[in], ic));
        wts.push_back(scalar.wts[in]);
      }
      offsets.push_back(dofs.size());
    }
  }

  HostCsrPattern pattern(num_obs,
                         num_states,
                         std::move(offsets),
                         std::move(dofs));
  HostCsrMatrix  mat(pattern);
  mat.vals() = std::move(wts);
  HostPointInterpolatorData data;
  data.mat_ = std::move(mat);
  return data;
}

} // namespace fem
} // namespace femx
