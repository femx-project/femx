#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/model/ns/NavierStokesModel.hpp>

namespace femx
{
using namespace fem;

namespace tests
{
namespace
{

TestOutcome modelOwnsReusableDiscretization()
{
  TestStatus status(__func__);

  model::ns::FluidParams fluid;
  fluid.rho = 1.2;
  fluid.mu  = 0.03;

  model::ns::NavierStokesModel model(
      Mesh::makeStructuredQuad(2, 1), 4, 0.05, fluid);

  status *= model.numSteps() == 4;
  status *= model.dt() == 0.05;
  status *= model.fluid().rho == 1.2;
  status *= model.fluid().mu == 0.03;
  status *= model.mesh().numElems() == 2;
  status *= model.numStates() == model.space().numDofs();

  const state::TimeDims dims  = model.residual().dims();
  status                     *= dims.num_steps == model.numSteps();
  status                     *= dims.num_states == model.numStates();
  status                     *= dims.num_param == 0;
  status                     *= dims.num_res == model.numStates();

  status *= model.map().graph().rows() == model.numStates();
  status *= model.map().graph().cols() == model.numStates();
  status *= model.map().numElems() == model.mesh().numElems();
  status *= model.velocityDofs().size()
            == model.mesh().numNodes() * model.mesh().dim();

  return status.report();
}

TestOutcome modelPublishesBackendAssemblyInputs()
{
  TestStatus status(__func__);

  model::ns::NavierStokesModel model(
      Mesh::makeStructuredQuad(2, 2), 2, 0.1, {});

  const auto&   geometry = model.geometry();
  const auto&   map      = model.map();
  HostCsrMatrix mat(map.graph());

  status *= geometry.dim() == model.mesh().dim();
  status *= geometry.numNodes() == model.mesh().numNodes();
  status *= geometry.numElems() == model.mesh().numElems();
  status *= map.numRes() == model.numStates();
  status *= map.numStates() == model.numStates();
  status *= mat.rows() == model.numStates();
  status *= mat.cols() == model.numStates();
  status *= mat.nnz() == map.graph().nnz();

  return status.report();
}

TestOutcome modelRejectsInvalidTimeConfiguration()
{
  TestStatus status(__func__);

  bool threw = false;
  try
  {
    model::ns::NavierStokesModel model(
        Mesh::makeStructuredQuad(1, 1), 0, 0.1, {});
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  threw = false;
  try
  {
    model::ns::NavierStokesModel model(
        Mesh::makeStructuredQuad(1, 1), 1, 0.0, {});
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::modelOwnsReusableDiscretization();
  results += femx::tests::modelPublishesBackendAssemblyInputs();
  results += femx::tests::modelRejectsInvalidTimeConfiguration();
  return results.summary();
}
