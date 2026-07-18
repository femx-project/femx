#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <type_traits>

#include "TestHelper.hpp"
#include <femx/assembly/Assembly.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>
#include <femx/model/ns/NavierStokesModel.hpp>

#if defined(FEMX_HAS_CUDA)
#include <femx/assembly/CudaAssembly.hpp>
#endif

namespace femx
{
namespace tests
{
namespace
{

using HostOp   = model::ns::NavierOperator<MemorySpace::Host>;
using DeviceOp = model::ns::NavierOperator<MemorySpace::Device>;

static_assert(std::is_trivially_copyable<HostOp>::value,
              "Host NavierOperator must be trivially copyable");
static_assert(std::is_trivially_copyable<DeviceOp>::value,
              "Device NavierOperator must be trivially copyable");

bool near(Real lhs, Real rhs, Real tol = 1.0e-11)
{
  return std::abs(lhs - rhs)
         <= tol * std::max<Real>({1.0, std::abs(lhs), std::abs(rhs)});
}

bool vecNear(const HostVector& lhs,
             const HostVector& rhs,
             Real              tol = 1.0e-11)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }
  for (Index i = 0; i < lhs.size(); ++i)
  {
    if (!near(lhs[i], rhs[i], tol))
    {
      std::cout << "    vector mismatch at " << i << ": " << lhs[i]
                << " != " << rhs[i] << '\n';
      return false;
    }
  }
  return true;
}

bool matNear(const HostCsrMatrix& lhs,
             const HostCsrMatrix& rhs,
             Real                 tol = 1.0e-11)
{
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()
      || lhs.nnz() != rhs.nnz())
  {
    return false;
  }
  for (Index row = 0; row <= lhs.rows(); ++row)
  {
    if (lhs.rowPtrData()[row] != rhs.rowPtrData()[row])
    {
      return false;
    }
  }
  for (Index k = 0; k < lhs.nnz(); ++k)
  {
    if (lhs.colIndData()[k] != rhs.colIndData()[k]
        || !near(lhs.valsData()[k], rhs.valsData()[k], tol))
    {
      std::cout << "    matrix mismatch at " << k << '\n';
      return false;
    }
  }
  return true;
}

HostVector apply(const HostCsrMatrix& mat, const HostVector& vec)
{
  HostVector out(mat.rows());
  for (Index row = 0; row < mat.rows(); ++row)
  {
    for (Index k = mat.rowPtrData()[row]; k < mat.rowPtrData()[row + 1]; ++k)
    {
      out[row] += mat.valsData()[k] * vec[mat.colIndData()[k]];
    }
  }
  return out;
}

Real dot(const HostVector& lhs, const HostVector& rhs)
{
  Real out = 0.0;
  for (Index i = 0; i < lhs.size(); ++i)
  {
    out += lhs[i] * rhs[i];
  }
  return out;
}

model::ns::NavierStokesModel makeModel()
{
  model::ns::FluidParams fluid;
  fluid.rho = 1.2;
  fluid.mu  = 0.03;
  return {fem::Mesh::makeStructuredQuad(2, 1), 3, 0.05, fluid};
}

void fillStates(Index num_states, HostVector& hist, HostVector& nxt)
{
  hist.resize(2 * num_states);
  nxt.resize(num_states);
  for (Index i = 0; i < num_states; ++i)
  {
    hist[i]              = 0.01 * (i + 1);
    hist[num_states + i] = -0.005 * (i + 1);
    nxt[i]               = 0.02 - 0.003 * i;
  }
}

TestOutcome navierDataFlattensEveryElement()
{
  TestStatus status(__func__);
  try
  {
    auto        model = makeModel();
    const auto  data  = model.data().view();
    const auto& vel   = model.space().field(0).space();
    const auto  quad  = fem::GaussQuadrature::make(
        vel.finiteElement().referenceElement(), 2);
    fem::ElementValues vals(vel.finiteElement(), quad);

    status *= data.numElems() == model.mesh().numElems();
    status *= data.numQpts() == quad.size();
    status *= data.numNodes() == vel.numShapesPerElem();
    status *= data.dim() == model.mesh().dim();
    status *= data.numDofs() == model.space().numDofsPerElem();

    for (Index ie = 0; ie < data.numElems(); ++ie)
    {
      vals.reinit(model.mesh().elem(ie));
      for (Index iq = 0; iq < data.numQpts(); ++iq)
      {
        status *= near(data.JxW(ie, iq), vals.JxW(iq));
        for (Index in = 0; in < data.numNodes(); ++in)
        {
          status *= near(data.N(iq, in), vals.N(iq)[in]);
          for (Index d = 0; d < data.dim(); ++d)
          {
            status *= near(data.dNdx(ie, iq, in, d), vals.dNdx(iq)(in, d));
          }
        }
      }
    }
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }
  return status.report();
}

TestOutcome navierModelResidualMatchesRowAssembly()
{
  TestStatus status(__func__);
  try
  {
    auto       model = makeModel();
    HostVector hist;
    HostVector nxt;
    fillStates(model.numStates(), hist, nxt);
    const HostVector         prm;
    const state::TimeContext time{
        1,
        &nxt,
        &prm,
        {hist.data(), 2, model.numStates()}};

    HostVector model_res;
    model.residual().res(time, model_res);
    linalg::MapCsrMatrix model_jac(model.map());
    status *= model.residual().assembleJac(
        time, state::VariableBlock::NextState, model_jac);
    DenseMatrix dense_jac;
    status *= model.residual().assembleJac(
        time, state::VariableBlock::NextState, dense_jac);

    HostVector    row_res;
    HostCsrMatrix row_jac(model.map().graph());
    CpuContext    ctx;
    assembly::assemble(model.op(),
                       1,
                       2,
                       state::VariableBlock::NextState,
                       model.map(),
                       hist,
                       nxt,
                       row_res,
                       row_jac,
                       ctx);

    status *= vecNear(row_res, model_res);
    status *= matNear(row_jac, model_jac.mat());

    HostVector sparse_vec;
    HostVector dense_vec;
    model_jac.apply(nxt, sparse_vec);
    dense_jac.apply(nxt, dense_vec);
    status *= vecNear(sparse_vec, dense_vec);

    HostVector    zero_nxt(model.numStates(), 0.0);
    HostVector    zero_res;
    HostCsrMatrix zero_jac(model.map().graph());
    assembly::assemble(model.op(),
                       1,
                       2,
                       state::VariableBlock::NextState,
                       model.map(),
                       hist,
                       zero_nxt,
                       zero_res,
                       zero_jac,
                       ctx);
    HostVector delta;
    model_jac.apply(nxt, delta);
    for (Index i = 0; i < delta.size(); ++i)
    {
      delta[i] += zero_res[i];
    }
    status *= vecNear(row_res, delta);
    status *= matNear(row_jac, zero_jac);
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }
  return status.report();
}

TestOutcome navierHistoryTangentsAreExact()
{
  TestStatus status(__func__);
  try
  {
    auto       model = makeModel();
    HostVector hist;
    HostVector nxt;
    fillStates(model.numStates(), hist, nxt);
    const HostVector         prm;
    const state::TimeContext time{
        1,
        &nxt,
        &prm,
        {hist.data(), 2, model.numStates()}};
    CpuContext ctx;

    HostVector dir(model.numStates());
    for (Index i = 0; i < dir.size(); ++i)
    {
      dir[i] = 0.2 - 0.01 * i;
    }

    for (Index lag = 0; lag < 2; ++lag)
    {
      const auto    wrt = state::VariableBlock::hist(lag);
      HostVector    res;
      HostCsrMatrix jac(model.map().graph());
      assembly::assemble(model.op(),
                         1,
                         2,
                         wrt,
                         model.map(),
                         hist,
                         nxt,
                         res,
                         jac,
                         ctx);

      linalg::MapCsrMatrix model_jac(model.map());
      status *= model.residual().assembleJac(time, wrt, model_jac);
      status *= matNear(jac, model_jac.mat());

      constexpr Real eps        = 1.0e-6;
      HostVector     plus_hist  = hist;
      HostVector     minus_hist = hist;
      for (Index i = 0; i < model.numStates(); ++i)
      {
        plus_hist[lag * model.numStates() + i]  += eps * dir[i];
        minus_hist[lag * model.numStates() + i] -= eps * dir[i];
      }

      HostVector    plus;
      HostVector    minus;
      HostCsrMatrix unused_plus(model.map().graph());
      HostCsrMatrix unused_minus(model.map().graph());
      assembly::assemble(model.op(),
                         1,
                         2,
                         state::VariableBlock::NextState,
                         model.map(),
                         plus_hist,
                         nxt,
                         plus,
                         unused_plus,
                         ctx);
      assembly::assemble(model.op(),
                         1,
                         2,
                         state::VariableBlock::NextState,
                         model.map(),
                         minus_hist,
                         nxt,
                         minus,
                         unused_minus,
                         ctx);

      const HostVector exact = apply(jac, dir);
      HostVector       fd(exact.size());
      for (Index i = 0; i < fd.size(); ++i)
      {
        fd[i] = (plus[i] - minus[i]) / (2.0 * eps);
      }
      status *= vecNear(exact, fd, 2.0e-6);
    }
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }
  return status.report();
}

TestOutcome navierBlockTransposeIdentity()
{
  TestStatus status(__func__);
  try
  {
    auto       model = makeModel();
    HostVector hist;
    HostVector nxt;
    fillStates(model.numStates(), hist, nxt);
    const HostVector         prm;
    const state::TimeContext time{
        1,
        &nxt,
        &prm,
        {hist.data(), 2, model.numStates()}};

    HostVector dir(model.numStates());
    HostVector adj(model.numStates());
    for (Index i = 0; i < model.numStates(); ++i)
    {
      dir[i] = 0.1 + 0.007 * i;
      adj[i] = -0.05 + 0.011 * i;
    }

    const state::VariableBlock blocks[] = {
        state::VariableBlock::NextState,
        state::VariableBlock::hist(0),
        state::VariableBlock::hist(1)};
    for (const auto wrt : blocks)
    {
      HostVector jv;
      HostVector jtw;
      model.residual().applyJac(time, wrt, dir, jv);
      model.residual().applyJacT(time, wrt, adj, jtw);
      status *= near(dot(jv, adj), dot(dir, jtw), 1.0e-10);
    }
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }
  return status.report();
}

TestOutcome navierRowAssemblyMatchesDevice()
{
  TestStatus status(__func__);
#if defined(FEMX_HAS_CUDA)
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    auto       model = makeModel();
    HostVector hist;
    HostVector nxt;
    fillStates(model.numStates(), hist, nxt);

    CpuContext cpu;

    CudaContext                 ctx;
    assembly::DeviceAssemblyMap map;
    model::ns::DeviceNavierData data;
    DeviceVector                dev_hist;
    DeviceVector                dev_nxt;
    assembly::copy(model.map(), map, ctx);
    model::ns::copy(model.data(), data, ctx);
    femx::copy(hist, dev_hist, ctx);
    femx::copy(nxt, dev_nxt, ctx);

    DeviceVector   dev_res;
    const DeviceOp op(
        data.view(),
        {model.fluid().rho, model.fluid().mu},
        model.dt());
    const state::VariableBlock blocks[] = {
        state::VariableBlock::NextState,
        state::VariableBlock::hist(0),
        state::VariableBlock::hist(1)};
    for (const auto wrt : blocks)
    {
      HostVector    host_res;
      HostCsrMatrix host_jac(model.map().graph());
      assembly::assemble(model.op(),
                         1,
                         2,
                         wrt,
                         model.map(),
                         hist,
                         nxt,
                         host_res,
                         host_jac,
                         cpu);

      DeviceCsrMatrix dev_jac(map.graph());
      assembly::assemble(op,
                         1,
                         2,
                         wrt,
                         map,
                         dev_hist,
                         dev_nxt,
                         dev_res,
                         dev_jac,
                         ctx);

      HostVector    got_res;
      HostCsrMatrix got_jac(model.map().graph());
      femx::copy(dev_res, got_res, ctx);
      femx::copy(dev_jac, got_jac, ctx);
      ctx.synchronize();
      status *= vecNear(got_res, host_res, 1.0e-9);
      status *= matNear(got_jac, host_jac, 1.0e-9);
    }
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }
#else
  status.skipTest();
#endif
  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::navierDataFlattensEveryElement();
  results += femx::tests::navierModelResidualMatchesRowAssembly();
  results += femx::tests::navierHistoryTangentsAreExact();
  results += femx::tests::navierBlockTransposeIdentity();
  results += femx::tests::navierRowAssemblyMatchesDevice();
  return results.summary();
}
