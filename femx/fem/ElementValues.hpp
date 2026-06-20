#pragma once

#include <vector>

#include <femx/core/Types.hpp>
#include <femx/algebra/MatrixView.hpp>
#include <femx/algebra/VectorView.hpp>

namespace femx
{

class Cell;
class FiniteElement;
class GaussQuadrature;

class ElementValues
{
public:
  ElementValues(const FiniteElement&   finite_element,
                const GaussQuadrature& quad);

  void reinit(const Cell& cell);

  Index numNodes() const;
  Index numDofs() const;
  Index dim() const;
  Index numQuadraturePoints() const;

  VectorView<const Real> N(Index iq) const;
  MatrixView<const Real> dNdr(Index iq) const;
  MatrixView<const Real> dNdx(Index iq) const;

  Real detJ(Index iq) const;
  Real weight(Index iq) const;
  Real JxW(Index iq) const;

  const Real* NData() const;
  const Real* dNdxData() const;
  const Real* JxWData() const;

private:
  void calcReferenceValues();
  void calcPhysicalValues(const Cell& cell);

  static Real invJacobian(const std::vector<Real>& J,
                          std::vector<Real>&       invJ,
                          Index                    dim);

private:
  const FiniteElement*   fe_{nullptr};
  const GaussQuadrature* quad_{nullptr};

  Index num_nodes_{0};
  Index num_dofs_{0};
  Index dim_{0};
  Index num_qp_{0};

  std::vector<Real> N_;
  std::vector<Real> dNdr_;
  std::vector<Real> dNdx_;

  std::vector<Real> detJ_;
  std::vector<Real> weights_;
  std::vector<Real> JxW_;

  std::vector<Real> J_;
  std::vector<Real> invJ_;
};

} // namespace femx
