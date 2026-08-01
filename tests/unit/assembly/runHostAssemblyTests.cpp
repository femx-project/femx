#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "TestHelper.hpp"
#include <femx/assembly/Assembly.hpp>
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/fem/DofMap.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/host/HostContext.hpp>
#include <femx/linalg/host/HostSystemMatrix.hpp>

namespace femx
{
namespace tests
{
namespace
{

bool near(Real lhs, Real rhs)
{
  return std::abs(lhs - rhs) <= 1.0e-12;
}

template <class Values, class T, std::size_t N>
bool valsEqual(const Values& vals, const std::array<T, N>& expected)
{
  if (vals.size() != static_cast<Index>(N))
  {
    return false;
  }
  for (Index i = 0; i < vals.size(); ++i)
  {
    if (vals[i] != expected[static_cast<std::size_t>(i)])
    {
      return false;
    }
  }
  return true;
}

template <class T, std::size_t N>
bool valsEqual(const T* vals, const std::array<T, N>& expected)
{
  for (std::size_t i = 0; i < N; ++i)
  {
    if (vals[i] != expected[i])
    {
      return false;
    }
  }
  return true;
}

Real csrVal(const HostCsrMatrix& mat, Index row, Index col)
{
  for (Index k = mat.rowPtrData()[row];
       k < mat.rowPtrData()[row + 1];
       ++k)
  {
    if (mat.colIndData()[k] == col)
    {
      return mat.valsData()[k];
    }
  }
  return 0.0;
}

struct AffineElementKernel
{
  void evalRow(const assembly::HostElementView& in,
               Index                            row,
               Real&                            res,
               HostVectorView<Real>             jac) const
  {
    res = in.state[row] + static_cast<Real>(in.ie + 1)
          + in.coords[0];
    for (Index col = 0; col < jac.size(); ++col)
    {
      jac[col] = row == col ? 2.0 : 1.0;
    }
  }
};

struct RectangularElementKernel
{
  void evalRow(const assembly::HostElementView& in,
               Index                            row,
               Real&                            res,
               HostVectorView<Real>             jac) const
  {
    res = static_cast<Real>(row + 1);
    for (Index col = 0; col < in.state.size(); ++col)
    {
      res      += in.state[col];
      jac[col]  = 10.0 * static_cast<Real>(row + 1)
                 + static_cast<Real>(col + 1);
    }
  }
};

struct TimeElementKernel
{
  template <MemorySpace Space>
  FEMX_HOST_DEVICE void evalRow(
      const assembly::TimeElementView<Space>& elem,
      state::VariableBlock                    wrt,
      Index                                   row,
      Real&                                   res,
      VectorView<Space, Real>                 jac) const
  {
    res = elem.nxt[row] - 2.0 * elem.histState(0)[row]
          + 0.5 * elem.histState(1)[row]
          + static_cast<Real>(elem.ie + elem.step);
    const Real diag = wrt.isNextState()
                          ? 1.0
                          : (wrt.historyLag() == 0 ? -2.0 : 0.5);
    for (Index col = 0; col < jac.size(); ++col)
    {
      jac[col] = row == col ? diag : 0.0;
    }
  }
};

TestOutcome meshProvidesRuntimeAssemblyData()
{
  TestStatus status(__func__);

  const fem::Mesh mesh = fem::Mesh::makeStructuredQuad(2, 1);

  status                          *= mesh.dim() == 2;
  status                          *= mesh.numNodes() == 6;
  status                          *= mesh.numElems() == 2;
  const auto                 view  = mesh.view();
  const std::array<Real, 12> coords{
      {0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0}};
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    for (Index id = 0; id < mesh.dim(); ++id)
    {
      status *=
          near(view.coord(in, id), coords[in * mesh.dim() + id]);
    }
  }
  const std::array<Index, 8> conn{{0, 1, 4, 3, 1, 2, 5, 4}};
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    status *= view.elemNumNodes(ie) == 4;
    for (Index in = 0; in < 4; ++in)
    {
      status *= view.elemNodeId(ie, in) == conn[ie * 4 + in];
    }
  }

  return status.report();
}

TestOutcome rectangularMapBuildsExactCsrMapping()
{
  TestStatus status(__func__);

  const fem::DofMap res_map(
      2,
      HostVector<Index>{0, 2, 3},
      HostVector<Index>{0, 1, 1});
  const fem::DofMap state_map(
      2,
      HostVector<Index>{0, 1, 3},
      HostVector<Index>{0, 0, 1});
  const auto map = assembly::makeAssemblyMap(res_map, state_map);

  status          *= map.numElems() == 2;
  status          *= map.pattern().nnz() == 3;
  const auto view  = map.view();
  status          *= valsEqual(view.res_offsets,
                      std::array<Index, 3>{{0, 2, 3}});
  status          *= valsEqual(view.state_offsets,
                      std::array<Index, 3>{{0, 1, 3}});
  status          *= valsEqual(view.jac_offsets,
                      std::array<Index, 3>{{0, 2, 4}});
  status          *= valsEqual(map.pattern().rowPtr(),
                      std::array<Index, 3>{{0, 1, 3}});
  status          *= valsEqual(map.pattern().colInd(),
                      std::array<Index, 3>{{0, 0, 1}});
  status          *= valsEqual(view.jac_map,
                      std::array<Index, 4>{{0, 1, 1, 2}});

  return status.report();
}

TestOutcome hostAssemblyUsesRuntimeMapAndSharedGraph()
{
  TestStatus status(__func__);

  fem::Mesh           mesh = fem::Mesh::makeStructuredQuad(2, 1);
  fem::LagrangeQuadQ1 element;
  fem::FESpace        space(&mesh, &element);
  space.setup();

  const auto map = assembly::makeAssemblyMap(space.dofMap());

  HostVector<Real>         res;
  const HostVector<Real>   state{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  linalg::HostContext      ctx;
  linalg::HostSystemMatrix jac_sys(ctx);

  jac_sys.setup(map.pattern());
  assembly::assembleResidualAndJacobian(AffineElementKernel{},
                                        mesh,
                                        map,
                                        state,
                                        res,
                                        jac_sys,
                                        ctx);
  const HostCsrMatrix& jac = jac_sys.matrix();

  status *= valsEqual(
      res,
      std::array<Real, 6>{{2.0, 7.5, 5.5, 5.0, 13.5, 8.5}});
  status *= jac.nnz() == 28;
  status *= near(csrVal(jac, 0, 0), 2.0);
  status *= near(csrVal(jac, 1, 1), 4.0);
  status *= near(csrVal(jac, 4, 4), 4.0);
  status *= near(csrVal(jac, 1, 4), 2.0);
  status *= near(csrVal(jac, 0, 2), 0.0);

  linalg::HostSystemMatrix jac_only(ctx);
  jac_only.setup(map.pattern());
  assembly::assembleJacobian(
      AffineElementKernel{}, mesh, map, state, jac_only, ctx);
  const HostCsrMatrix& jac_only_mat  = jac_only.matrix();
  status                            *= jac_only_mat.nnz() == jac.nnz();
  for (Index entry = 0; entry < jac.nnz(); ++entry)
  {
    status *=
        near(jac_only_mat.valsData()[entry], jac.valsData()[entry]);
  }

  return status.report();
}

TestOutcome hostAssemblySupportsRectangularLocalLayouts()
{
  TestStatus status(__func__);

  const fem::Mesh                     mesh = fem::Mesh::makeStructuredQuad(2, 1);
  const HostVector<HostVector<Index>> res_dofs{{0, 1}, {1}};
  const HostVector<HostVector<Index>> state_dofs{{0}, {0, 1}};
  const auto                          map =
      assembly::makeAssemblyMap(2, 2, res_dofs, state_dofs);

  const HostVector<Real>   state{2.0, 3.0};
  HostVector<Real>         res;
  linalg::HostContext      ctx;
  linalg::HostSystemMatrix jac_sys(ctx);
  jac_sys.setup(map.pattern());
  assembly::assembleResidualAndJacobian(RectangularElementKernel{},
                                        mesh,
                                        map,
                                        state,
                                        res,
                                        jac_sys,
                                        ctx);
  const HostCsrMatrix& jac = jac_sys.matrix();

  status *= valsEqual(res, std::array<Real, 2>{{3.0, 10.0}});
  status *= near(csrVal(jac, 0, 0), 11.0);
  status *= near(csrVal(jac, 1, 0), 32.0);
  status *= near(csrVal(jac, 1, 1), 12.0);

  return status.report();
}

TestOutcome hostTimeAssemblyHandlesHistoryBlocks()
{
  TestStatus status(__func__);

  const auto map = assembly::makeAssemblyMap(
      3,
      3,
      HostVector<HostVector<Index>>{{0, 1}, {1, 2}},
      HostVector<HostVector<Index>>{{0, 1}, {1, 2}});
  const HostVector<Real>   hist{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const HostVector<Real>   nxt{7.0, 8.0, 9.0};
  HostVector<Real>         res;
  linalg::HostContext      ctx;
  linalg::HostSystemMatrix jac_sys(ctx);

  jac_sys.setup(map.pattern());
  assembly::assembleResidualAndJacobian(TimeElementKernel{},
                                        3,
                                        2,
                                        state::VariableBlock::NextState,
                                        map,
                                        0,
                                        map.numElems(),
                                        hist.view(),
                                        nxt.view(),
                                        res,
                                        jac_sys,
                                        ctx);
  const HostCsrMatrix& jac  = jac_sys.matrix();
  status                   *= valsEqual(res, std::array<Real, 3>{{10.0, 20.0, 10.0}});
  status                   *= near(csrVal(jac, 0, 0), 1.0);
  status                   *= near(csrVal(jac, 1, 1), 2.0);
  status                   *= near(csrVal(jac, 2, 2), 1.0);

  jac_sys.setup(map.pattern());
  assembly::assembleResidualAndJacobian(TimeElementKernel{},
                                        3,
                                        2,
                                        state::VariableBlock::hist(0),
                                        map,
                                        0,
                                        map.numElems(),
                                        hist.view(),
                                        nxt.view(),
                                        res,
                                        jac_sys,
                                        ctx);
  status *= near(csrVal(jac, 0, 0), -2.0);
  status *= near(csrVal(jac, 1, 1), -4.0);
  status *= near(csrVal(jac, 2, 2), -2.0);

  return status.report();
}

TestOutcome hostTimeAssemblySupportsElementRangesAndResidualOnly()
{
  TestStatus status(__func__);

  const auto map = assembly::makeAssemblyMap(
      3,
      3,
      HostVector<HostVector<Index>>{{0, 1}, {1, 2}},
      HostVector<HostVector<Index>>{{0, 1}, {1, 2}});
  const HostVector<Real>   hist{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const HostVector<Real>   nxt{7.0, 8.0, 9.0};
  HostVector<Real>         res;
  linalg::HostContext      ctx;
  linalg::HostSystemMatrix jac_sys(ctx);

  jac_sys.setup(map.pattern());
  assembly::assembleResidualAndJacobian(TimeElementKernel{},
                                        3,
                                        2,
                                        state::VariableBlock::NextState,
                                        map,
                                        0,
                                        1,
                                        hist.view(),
                                        nxt.view(),
                                        res,
                                        jac_sys,
                                        ctx);
  const HostCsrMatrix& jac  = jac_sys.matrix();
  status                   *= valsEqual(res, std::array<Real, 3>{{10.0, 9.5, 0.0}});
  status                   *= near(csrVal(jac, 0, 0), 1.0);
  status                   *= near(csrVal(jac, 1, 1), 1.0);
  status                   *= near(csrVal(jac, 2, 2), 0.0);

  assembly::assembleResidual(TimeElementKernel{},
                             3,
                             2,
                             map,
                             0,
                             1,
                             hist.view(),
                             nxt.view(),
                             res,
                             ctx);
  status *= valsEqual(res, std::array<Real, 3>{{10.0, 9.5, 0.0}});

  assembly::assembleResidual(TimeElementKernel{},
                             3,
                             2,
                             map,
                             0,
                             map.numElems(),
                             hist.view(),
                             nxt.view(),
                             res,
                             ctx);
  status *= valsEqual(res, std::array<Real, 3>{{10.0, 20.0, 10.0}});

  return status.report();
}

TestOutcome matGraphSurvivesAssemblyMapMove()
{
  TestStatus status(__func__);

  const fem::Mesh mesh = fem::Mesh::makeStructuredQuad(1, 1);
  auto            map  = assembly::makeAssemblyMap(
      4,
      4,
      HostVector<HostVector<Index>>{{0, 1, 2, 3}},
      HostVector<HostVector<Index>>{{0, 1, 2, 3}});
  auto moved_map = std::move(map);

  HostVector<Real>         res;
  const HostVector<Real>   state{1.0, 2.0, 3.0, 4.0};
  linalg::HostContext      ctx;
  linalg::HostSystemMatrix jac_sys(ctx);
  jac_sys.setup(moved_map.pattern());
  assembly::assembleResidualAndJacobian(AffineElementKernel{},
                                        mesh,
                                        moved_map,
                                        state,
                                        res,
                                        jac_sys,
                                        ctx);
  const HostCsrMatrix& jac = jac_sys.matrix();

  status *= jac.rows() == 4;
  status *= jac.nnz() == 16;
  status *= near(csrVal(jac, 0, 0), 2.0);
  status *= near(csrVal(jac, 3, 2), 1.0);

  return status.report();
}

TestOutcome malformedGraphsAndAssemblyAliasesAreRejected()
{
  TestStatus status(__func__);

  bool malformed_rejected = false;
  try
  {
    HostCsrPattern invalid(2,
                           2,
                           HostVector<Index>{0, 2, 1},
                           HostVector<Index>{0});
    (void) invalid;
  }
  catch (const std::runtime_error&)
  {
    malformed_rejected = true;
  }
  status *= malformed_rejected;

  fem::Mesh           mesh = fem::Mesh::makeStructuredQuad(1, 1);
  fem::LagrangeQuadQ1 element;
  fem::FESpace        space(&mesh, &element);
  space.setup();
  const auto               map = assembly::makeAssemblyMap(space.dofMap());
  HostVector<Real>         alias_vec{1.0, 2.0, 3.0, 4.0};
  linalg::HostContext      ctx;
  linalg::HostSystemMatrix jac_sys(ctx);
  jac_sys.setup(map.pattern());

  bool alias_rejected = false;
  try
  {
    assembly::assembleResidualAndJacobian(AffineElementKernel{},
                                          mesh,
                                          map,
                                          alias_vec,
                                          alias_vec,
                                          jac_sys,
                                          ctx);
  }
  catch (const std::runtime_error&)
  {
    alias_rejected = true;
  }
  status *= alias_rejected;

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::meshProvidesRuntimeAssemblyData();
  results += femx::tests::rectangularMapBuildsExactCsrMapping();
  results += femx::tests::hostAssemblyUsesRuntimeMapAndSharedGraph();
  results += femx::tests::hostAssemblySupportsRectangularLocalLayouts();
  results += femx::tests::hostTimeAssemblyHandlesHistoryBlocks();
  results +=
      femx::tests::hostTimeAssemblySupportsElementRangesAndResidualOnly();
  results += femx::tests::matGraphSurvivesAssemblyMapMove();
  results += femx::tests::malformedGraphsAndAssemblyAliasesAreRejected();
  return results.summary();
}
