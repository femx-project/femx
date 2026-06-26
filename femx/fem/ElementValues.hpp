#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/MatrixView.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{

class Element;
class FiniteElement;
class GaussQuadrature;

class ElementValues
{
public:
  ElementValues(const FiniteElement&   finite_element,
                const GaussQuadrature& quad);

  void reinit(const Element& elem);

  Index numNodes() const;
  Index numDofs() const;
  Index dim() const;
  Index numQuadraturePoints() const;

  VectorView<const Real> N(Index iq) const;
  MatrixView<const Real> dNdr(Index iq) const;
  MatrixView<const Real> dNdx(Index iq) const;

  Real detJ(Index iq) const;
  Real wt(Index iq) const;
  Real JxW(Index iq) const;

  const Real* NData() const;
  const Real* dNdxData() const;
  const Real* JxWData() const;

private:
  void calcReferenceValues();
  void calcPhysicalValues(const Element& elem);

  static Real invJacobian(const Vector<Real>& J,
                          Vector<Real>&       invJ,
                          Index               dim);

private:
  const FiniteElement*   fe_{nullptr};
  const GaussQuadrature* quad_{nullptr};

  Index nn_{0};
  Index nd_{0};
  Index dim_{0};
  Index nq_{0};

  Vector<Real> N_;
  Vector<Real> dNdr_;
  Vector<Real> dNdx_;

  Vector<Real> detJ_;
  Vector<Real> wts_;
  Vector<Real> JxW_;

  Vector<Real> J_;
  Vector<Real> invJ_;
};

} // namespace femx
