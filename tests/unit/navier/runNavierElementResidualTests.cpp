#include <iostream>
#include <vector>

#include "Assembly.hpp"
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class NavierElementResidualTests : public TestBase
{
public:
  TestOutcome residualMatchesElementSystem()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 elem;

    FESpace u_space(&mesh, &elem, mesh.dim());
    FESpace p_space(&mesh, &elem);

    MixedFESpace space;
    space.addField(u_space);
    space.addField(p_space);
    space.setup();

    Vector<Real> x_next(space.numDofs());
    Vector<Real> x(space.numDofs());
    Vector<Real> xp(space.numDofs());
    for (Index i = 0; i < space.numDofs(); ++i)
    {
      x_next[i] = 0.11 + 0.013 * static_cast<Real>(i);
      x[i]      = 0.07 + 0.017 * static_cast<Real>(i);
      xp[i]     = 0.03 + 0.019 * static_cast<Real>(i);
    }

    FluidParams fluid;
    fluid.rho = 1.2;
    fluid.mu  = 0.03;

    const Real dt   = 0.2;
    const auto quad = GaussQuadrature::make(elem.referenceElement(), 2);

    ElementValues        ev(elem, quad);
    std::vector<QPState> qp(
        static_cast<std::size_t>(quad.size()));
    DenseMatrix  Ke(space.numDofsPerElem(), space.numDofsPerElem());
    Vector<Real> Fe(space.numDofsPerElem());

    Real max_cfl = 0.0;
    assembleElemSystem(space,
                       0,
                       ev,
                       qp,
                       x,
                       xp,
                       false,
                       fluid,
                       dt,
                       Ke,
                       Fe,
                       max_cfl);

    Vector<Real> Re;
    elemResidualFromSystem(space, 0, Ke, Fe, x_next, Re);

    const Vector<Index> dofs = space.elemDofs(0);
    for (Index i = 0; i < space.numDofsPerElem(); ++i)
    {
      Real expected = -Fe[i];
      for (Index j = 0; j < space.numDofsPerElem(); ++j)
      {
        expected += Ke(i, j) * x_next[dofs[j]];
      }

      if (!isEqual(Re[i], expected))
      {
        std::cout << "Incorrect residual entry " << i << ": expected "
                  << expected << ", got " << Re[i] << "\n";
        status = false;
      }
    }

    Vector<Real> assembled_res;
    Real         assembled_max_cfl = 0.0;
    assembleElemResidual(space,
                         0,
                         ev,
                         qp,
                         x_next,
                         x,
                         xp,
                         false,
                         fluid,
                         dt,
                         assembled_res,
                         assembled_max_cfl);

    for (Index i = 0; i < space.numDofsPerElem(); ++i)
    {
      if (!isEqual(assembled_res[i], Re[i]))
      {
        std::cout << "assembleElemResidual mismatch at entry " << i
                  << ": expected " << Re[i] << ", got "
                  << assembled_res[i] << "\n";
        status = false;
      }
    }

    if (!isEqual(assembled_max_cfl, max_cfl))
    {
      std::cout << "assembleElemResidual max CFL mismatch: expected "
                << max_cfl << ", got " << assembled_max_cfl << "\n";
      status = false;
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running Navier elem residual tests:\n";

  femx::tests::NavierElementResidualTests test;

  femx::tests::TestingResults result;
  result += test.residualMatchesElementSystem();

  return result.summary();
}
