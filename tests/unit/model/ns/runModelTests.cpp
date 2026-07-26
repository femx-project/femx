#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/model/ns/Model.hpp>
#include <femx/model/ns/NavierStokesResidual.hpp>

namespace femx
{
using namespace fem;

namespace tests
{
namespace
{

TestOutcome ownsReusableDiscretization()
{
  TestStatus status(__func__);

  model::ns::FluidProperties fluid;
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

  model::ns::HostNavierStokesResidual res(model);
  const state::TimeDims               dims  = res.dims();
  status                                   *= dims.num_steps == model.numSteps();
  status                                   *= dims.num_states == model.numStates();
  status                                   *= dims.num_param == 0;
  status                                   *= dims.num_res == model.numStates();

  status *= model.assemblyMap().pattern().rows() == model.numStates();
  status *= model.assemblyMap().pattern().cols() == model.numStates();
  status *= model.assemblyMap().numElems() == model.mesh().numElems();
  status *= model.velocityDofs().size()
            == model.mesh().numNodes() * model.mesh().dim();

  return status.report();
}

TestOutcome modelPublishesAssemblyInputs()
{
  TestStatus status(__func__);

  model::ns::NavierStokesModel model(
      Mesh::makeStructuredQuad(2, 2), 2, 0.1, {});

  const auto&   mesh = model.mesh();
  const auto&   map  = model.assemblyMap();
  HostCsrMatrix mat(map.pattern());

  status *= mesh.dim() == 2;
  status *= mesh.numNodes() == 9;
  status *= mesh.numElems() == 4;
  status *= map.numRes() == model.numStates();
  status *= map.numStates() == model.numStates();
  status *= mat.rows() == model.numStates();
  status *= mat.cols() == model.numStates();
  status *= mat.nnz() == map.pattern().nnz();

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
  results += femx::tests::ownsReusableDiscretization();
  results += femx::tests::modelPublishesAssemblyInputs();
  results += femx::tests::modelRejectsInvalidTimeConfiguration();
  return results.summary();
}
