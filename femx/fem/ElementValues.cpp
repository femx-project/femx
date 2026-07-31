
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <femx/common/Checks.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>

namespace femx
{
namespace fem
{

ElementValues::ElementValues(const FiniteElement&   fe,
                             const GaussQuadrature& quad)
  : fe_(&fe),
    quad_(&quad),
    num_nodes_(fe.numNodes()),
    num_dofs_(fe.numDofsPerElement()),
    dim_(fe.dim()),
    num_qpts_(quad.size())
{
  N_.resize(num_qpts_ * num_dofs_);
  dNdr_.resize(num_qpts_ * num_dofs_ * dim_);
  dNdx_.resize(num_qpts_ * num_dofs_ * dim_);

  detJ_.resize(num_qpts_);
  wts_.resize(num_qpts_);
  JxW_.resize(num_qpts_);

  J_.resize(dim_ * dim_);
  invJ_.resize(dim_ * dim_);

  calcReferenceValues();
}

void ElementValues::reinit(const Mesh& mesh, Index ie)
{
  require(mesh.elemNumNodes(ie) == num_nodes_,
          "ElementValues mesh element node count does not match finite element");
  calcPhysicalValues(mesh, ie);
}

Index ElementValues::numNodes() const
{
  return num_nodes_;
}

Index ElementValues::numDofs() const
{
  return num_dofs_;
}

Index ElementValues::dim() const
{
  return dim_;
}

Index ElementValues::numQuadraturePoints() const
{
  return num_qpts_;
}

HostVectorView<const Real> ElementValues::N(Index iq) const
{
  return HostVectorView<const Real>(
      N_.data() + iq * num_dofs_,
      num_dofs_);
}

HostMatrixView<const Real> ElementValues::dNdr(Index iq) const
{
  return HostMatrixView<const Real>(
      dNdr_.data() + iq * num_dofs_ * dim_,
      num_dofs_,
      dim_);
}

HostMatrixView<const Real> ElementValues::dNdx(Index iq) const
{
  return HostMatrixView<const Real>(
      dNdx_.data() + iq * num_dofs_ * dim_,
      num_dofs_,
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
  for (Index iq = 0; iq < num_qpts_; ++iq)
  {
    const QuadraturePoint& qp = (*quad_)[iq];

    HostVectorView<Real> N(
        N_.data() + iq * num_dofs_,
        num_dofs_);

    HostMatrixView<Real> dNdr(
        dNdr_.data() + iq * num_dofs_ * dim_,
        num_dofs_,
        dim_);

    fe_->calcN(qp, N);
    fe_->calcdNdr(qp, dNdr);

    wts_[iq] = qp.wt;
  }
}

void ElementValues::calcPhysicalValues(const Mesh& mesh, Index ie)
{
  for (Index iq = 0; iq < num_qpts_; ++iq)
  {
    const auto dNdr_iq = dNdr(iq);

    std::fill(J_.begin(), J_.end(), 0.0);

    for (Index in = 0; in < num_nodes_; ++in)
    {
      for (Index id = 0; id < dim_; ++id)
      {
        const Real x = mesh.elemNode(ie, in)[id];

        for (Index jd = 0; jd < dim_; ++jd)
        {
          J_[id * dim_ + jd] += x * dNdr_iq(in, jd);
        }
      }
    }

    const Real detJ = invJacobian(J_, invJ_, dim_);
    detJ_[iq]       = std::abs(detJ);
    JxW_[iq]        = detJ_[iq] * wts_[iq];

    HostMatrixView<Real> dNdx(
        dNdx_.data() + iq * num_dofs_ * dim_,
        num_dofs_,
        dim_);

    for (Index in = 0; in < num_dofs_; ++in)
    {
      for (Index id = 0; id < dim_; ++id)
      {
        dNdx(in, id) = 0.0;

        for (Index jd = 0; jd < dim_; ++jd)
        {
          dNdx(in, id) +=
              dNdr_iq(in, jd) * invJ_[jd * dim_ + id];
        }
      }
    }
  }
}

Real ElementValues::invJacobian(const HostVector<Real>& J,
                                HostVector<Real>&       invJ,
                                Index                   dim)
{
  constexpr Real eps = 1.0e-30;

  if (dim == 1)
  {
    const Real det = J[0];

    if (std::abs(det) < eps)
    {
      throw std::runtime_error("Singular 1D Jacobian.");
    }

    invJ[0] = 1.0 / det;

    return det;
  }

  if (dim == 2)
  {
    const Real det = J[0] * J[3] - J[1] * J[2];

    if (std::abs(det) < eps)
    {
      throw std::runtime_error("Singular 2D Jacobian.");
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

    if (std::abs(det) < eps)
    {
      throw std::runtime_error("Singular 3D Jacobian.");
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

  throw std::runtime_error("Unsupported Jacobian dimension.");
}

} // namespace fem
} // namespace femx
