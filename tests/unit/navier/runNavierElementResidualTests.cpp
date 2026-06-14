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

    Vector x_next(space.numDofs());
    Vector x(space.numDofs());
    Vector xp(space.numDofs());
    for (index_type i = 0; i < space.numDofs(); ++i)
    {
      x_next[i] = 0.11 + 0.013 * static_cast<real_type>(i);
      x[i]      = 0.07 + 0.017 * static_cast<real_type>(i);
      xp[i]     = 0.03 + 0.019 * static_cast<real_type>(i);
    }

    FluidParams fluid;
    fluid.rho = 1.2;
    fluid.mu  = 0.03;

    const real_type dt   = 0.2;
    const auto      quad = GaussQuadrature::make(elem.referenceElement(), 2);

    ElementValues        ev(elem, quad);
    std::vector<QPState> qp(
        static_cast<std::size_t>(quad.size()));
    DenseMatrix Ke(space.numDofsPerElem(), space.numDofsPerElem());
    Vector      Fe(space.numDofsPerElem());

    real_type max_cfl = 0.0;
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

    Vector Re;
    elemResidualFromSystem(space, 0, Ke, Fe, x_next, Re);

    const std::vector<index_type> dofs = space.elemDofs(0);
    for (index_type i = 0; i < space.numDofsPerElem(); ++i)
    {
      real_type expected = -Fe[i];
      for (index_type j = 0; j < space.numDofsPerElem(); ++j)
      {
        expected += Ke(i, j) * x_next[dofs[static_cast<std::size_t>(j)]];
      }

      if (!isEqual(Re[i], expected))
      {
        std::cout << "Incorrect residual entry " << i << ": expected "
                  << expected << ", got " << Re[i] << "\n";
        status = false;
      }
    }

    Vector    assembled_residual;
    real_type assembled_max_cfl = 0.0;
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
                         assembled_residual,
                         assembled_max_cfl);

    for (index_type i = 0; i < space.numDofsPerElem(); ++i)
    {
      if (!isEqual(assembled_residual[i], Re[i]))
      {
        std::cout << "assembleElemResidual mismatch at entry " << i
                  << ": expected " << Re[i] << ", got "
                  << assembled_residual[i] << "\n";
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
