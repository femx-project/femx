
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <femx/fem/Cell.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FiniteElement.hpp>

using namespace std;

namespace femx
{

ElementValues::ElementValues(const FiniteElement&   fe,
                             const GaussQuadrature& quad)
  : fe_(&fe),
    quad_(&quad),
    nn_(fe.numNodes()),
    nd_(fe.numDofsPerElement()),
    dim_(fe.dim()),
    nq_(quad.size())
{
  N_.resize(nq_ * nd_);
  dNdr_.resize(nq_ * nd_ * dim_);
  dNdx_.resize(nq_ * nd_ * dim_);

  detJ_.resize(nq_);
  wts_.resize(nq_);
  JxW_.resize(nq_);

  J_.resize(dim_ * dim_);
  invJ_.resize(dim_ * dim_);

  calcReferenceValues();
}

void ElementValues::reinit(const Cell& cell)
{
  calcPhysicalValues(cell);
}

Index ElementValues::numNodes() const
{
  return nn_;
}

Index ElementValues::numDofs() const
{
  return nd_;
}

Index ElementValues::dim() const
{
  return dim_;
}

Index ElementValues::numQuadraturePoints() const
{
  return nq_;
}

VectorView<const Real> ElementValues::N(Index iq) const
{
  return VectorView<const Real>(
      N_.data() + iq * nd_,
      nd_);
}

MatrixView<const Real> ElementValues::dNdr(Index iq) const
{
  return MatrixView<const Real>(
      dNdr_.data() + iq * nd_ * dim_,
      nd_,
      dim_);
}

MatrixView<const Real> ElementValues::dNdx(Index iq) const
{
  return MatrixView<const Real>(
      dNdx_.data() + iq * nd_ * dim_,
      nd_,
      dim_);
}

Real ElementValues::detJ(Index iq) const
{
  return detJ_[iq];
}

Real ElementValues::wt(Index iq) const
{
  return wts_[iq];
}

Real ElementValues::JxW(Index iq) const
{
  return JxW_[iq];
}

const Real* ElementValues::NData() const
{
  return N_.data();
}

const Real* ElementValues::dNdxData() const
{
  return dNdx_.data();
}

const Real* ElementValues::JxWData() const
{
  return JxW_.data();
}

void ElementValues::calcReferenceValues()
{
  for (Index iq = 0; iq < nq_; ++iq)
  {
    const QuadraturePoint& qp = (*quad_)[iq];

    VectorView<Real> N(
        N_.data() + iq * nd_,
        nd_);

    MatrixView<Real> dNdr(
        dNdr_.data() + iq * nd_ * dim_,
        nd_,
        dim_);

    fe_->calcShape(qp, N);
    fe_->calcShapeGrad(qp, dNdr);

    wts_[iq] = qp.wt;
  }
}

void ElementValues::calcPhysicalValues(const Cell& cell)
{
  for (Index iq = 0; iq < nq_; ++iq)
  {
    const auto dNdr_iq = dNdr(iq);

    fill(J_.begin(), J_.end(), 0.0);

    for (Index in = 0; in < nn_; ++in)
    {
      for (Index a = 0; a < dim_; ++a)
      {
        const Real x_a = cell.node(in)[a];

        for (Index b = 0; b < dim_; ++b)
        {
          J_[a * dim_ + b] += x_a * dNdr_iq(in, b);
        }
      }
    }

    const Real detJ = invJacobian(J_, invJ_, dim_);
    detJ_[iq]       = abs(detJ);
    JxW_[iq]        = detJ_[iq] * wts_[iq];

    MatrixView<Real> dNdx(
        dNdx_.data() + iq * nd_ * dim_,
        nd_,
        dim_);

    for (Index i = 0; i < nd_; ++i)
    {
      for (Index a = 0; a < dim_; ++a)
      {
        dNdx(i, a) = 0.0;

        for (Index b = 0; b < dim_; ++b)
        {
          dNdx(i, a) += dNdr_iq(i, b) * invJ_[b * dim_ + a];
        }
      }
    }
  }
}

Real ElementValues::invJacobian(const Vector<Real>& J,
                                Vector<Real>&       invJ,
                                Index               dim)
{
  constexpr Real eps = 1.0e-30;

  if (dim == 1)
  {
    const Real det = J[0];

    if (abs(det) < eps)
    {
      throw runtime_error("Singular 1D Jacobian.");
    }

    invJ[0] = 1.0 / det;

    return det;
  }

  if (dim == 2)
  {
    const Real det = J[0] * J[3] - J[1] * J[2];

    if (abs(det) < eps)
    {
      throw runtime_error("Singular 2D Jacobian.");
    }

    const Real inv_det = 1.0 / det;

    invJ[0] = J[3] * inv_det;
    invJ[1] = -J[1] * inv_det;
    invJ[2] = -J[2] * inv_det;
    invJ[3] = J[0] * inv_det;

    return det;
  }

  if (dim == 3)
  {
    const Real det =
        J[0] * (J[4] * J[8] - J[5] * J[7])
        - J[1] * (J[3] * J[8] - J[5] * J[6])
        + J[2] * (J[3] * J[7] - J[4] * J[6]);

    if (abs(det) < eps)
    {
      throw runtime_error("Singular 3D Jacobian.");
    }

    const Real inv_det = 1.0 / det;

    invJ[0] = (J[4] * J[8] - J[5] * J[7]) * inv_det;
    invJ[1] = -(J[1] * J[8] - J[2] * J[7]) * inv_det;
    invJ[2] = (J[1] * J[5] - J[2] * J[4]) * inv_det;

    invJ[3] = -(J[3] * J[8] - J[5] * J[6]) * inv_det;
    invJ[4] = (J[0] * J[8] - J[2] * J[6]) * inv_det;
    invJ[5] = -(J[0] * J[5] - J[2] * J[3]) * inv_det;

    invJ[6] = (J[3] * J[7] - J[4] * J[6]) * inv_det;
    invJ[7] = -(J[0] * J[7] - J[1] * J[6]) * inv_det;
    invJ[8] = (J[0] * J[4] - J[1] * J[3]) * inv_det;

    return det;
  }

  throw runtime_error("Unsupported Jacobian dimension.");
}

} // namespace femx
