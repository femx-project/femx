
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <refem/fe/ElementValues.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/mesh/Cell.hpp>

namespace refem
{

ElementValues::ElementValues(const FiniteElement&   finite_element,
                             const GaussQuadrature& quadrature)
  : finite_element_(&finite_element),
    quadrature_(&quadrature),
    num_nodes_(finite_element.numNodes()),
    num_dofs_(finite_element.numDofsPerElement()),
    dim_(finite_element.dim()),
    num_qp_(quadrature.size())
{
  N_.resize(num_qp_ * num_dofs_);
  dNdr_.resize(num_qp_ * num_dofs_ * dim_);
  dNdx_.resize(num_qp_ * num_dofs_ * dim_);

  detJ_.resize(num_qp_);
  weights_.resize(num_qp_);

  J_.resize(dim_ * dim_);
  invJ_.resize(dim_ * dim_);

  calcReferenceValues();
}

void ElementValues::reinit(const Cell& cell)
{
  calcPhysicalValues(cell);
}

index_type ElementValues::numNodes() const
{
  return num_nodes_;
}

index_type ElementValues::numDofs() const
{
  return num_dofs_;
}

index_type ElementValues::dim() const
{
  return dim_;
}

index_type ElementValues::numQuadraturePoints() const
{
  return num_qp_;
}

VectorView<const real_type> ElementValues::N(index_type q) const
{
  return VectorView<const real_type>(
      N_.data() + q * num_dofs_,
      num_dofs_);
}

MatrixView<const real_type> ElementValues::dNdr(index_type q) const
{
  return MatrixView<const real_type>(
      dNdr_.data() + q * num_dofs_ * dim_,
      num_dofs_,
      dim_);
}

MatrixView<const real_type> ElementValues::dNdx(index_type q) const
{
  return MatrixView<const real_type>(
      dNdx_.data() + q * num_dofs_ * dim_,
      num_dofs_,
      dim_);
}

real_type ElementValues::detJ(index_type q) const
{
  return detJ_[q];
}

real_type ElementValues::weight(index_type q) const
{
  return weights_[q];
}

real_type ElementValues::JxW(index_type q) const
{
  return detJ_[q] * weights_[q];
}

void ElementValues::calcReferenceValues()
{
  for (index_type q = 0; q < num_qp_; ++q)
  {
    const QuadraturePoint& qp = (*quadrature_)[q];

    VectorView<real_type> N(
        N_.data() + q * num_dofs_,
        num_dofs_);

    MatrixView<real_type> dNdr(
        dNdr_.data() + q * num_dofs_ * dim_,
        num_dofs_,
        dim_);

    finite_element_->calcShape(qp, N);
    finite_element_->calcShapeGrad(qp, dNdr);

    weights_[q] = qp.weight;
  }
}

void ElementValues::calcPhysicalValues(const Cell& cell)
{
  for (index_type q = 0; q < num_qp_; ++q)
  {
    const auto dNdr_q = dNdr(q);

    std::fill(J_.begin(), J_.end(), 0.0);

    for (index_type i = 0; i < num_nodes_; ++i)
    {
      for (index_type a = 0; a < dim_; ++a)
      {
        const real_type x_a = cell.node(i)[a];

        for (index_type b = 0; b < dim_; ++b)
        {
          J_[a * dim_ + b] += x_a * dNdr_q(i, b);
        }
      }
    }

    const real_type detJ = invJacobian(J_, invJ_, dim_);
    detJ_[q]             = std::abs(detJ);

    MatrixView<real_type> dNdx(
        dNdx_.data() + q * num_dofs_ * dim_,
        num_dofs_,
        dim_);

    for (index_type i = 0; i < num_dofs_; ++i)
    {
      for (index_type a = 0; a < dim_; ++a)
      {
        dNdx(i, a) = 0.0;

        for (index_type b = 0; b < dim_; ++b)
        {
          dNdx(i, a) += dNdr_q(i, b) * invJ_[b * dim_ + a];
        }
      }
    }
  }
}

real_type ElementValues::invJacobian(const std::vector<real_type>& J,
                                     std::vector<real_type>&       invJ,
                                     index_type                    dim)
{
  constexpr real_type eps = 1.0e-30;

  if (dim == 1)
  {
    const real_type det = J[0];

    if (std::abs(det) < eps)
    {
      throw std::runtime_error("Singular 1D Jacobian.");
    }

    invJ[0] = 1.0 / det;

    return det;
  }

  if (dim == 2)
  {
    const real_type det = J[0] * J[3] - J[1] * J[2];

    if (std::abs(det) < eps)
    {
      throw std::runtime_error("Singular 2D Jacobian.");
    }

    const real_type inv_det = 1.0 / det;

    invJ[0] = J[3] * inv_det;
    invJ[1] = -J[1] * inv_det;
    invJ[2] = -J[2] * inv_det;
    invJ[3] = J[0] * inv_det;

    return det;
  }

  if (dim == 3)
  {
    const real_type det =
        J[0] * (J[4] * J[8] - J[5] * J[7])
        - J[1] * (J[3] * J[8] - J[5] * J[6])
        + J[2] * (J[3] * J[7] - J[4] * J[6]);

    if (std::abs(det) < eps)
    {
      throw std::runtime_error("Singular 3D Jacobian.");
    }

    const real_type inv_det = 1.0 / det;

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

} // namespace refem
