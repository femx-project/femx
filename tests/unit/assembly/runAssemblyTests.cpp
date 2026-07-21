#include <array>
#include <cmath>

#include "TestHelper.hpp"
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx
{
using namespace fem;

namespace tests
{
namespace
{

class IdentityTimeResidual final : public state::HostTimeResidual
{
public:
  IdentityTimeResidual()
    : pattern_(assembly::makeAssemblyMap(
                   3,
                   3,
                   Array<Array<Index>>{{0, 1, 2}},
                   Array<Array<Index>>{{0, 1, 2}})
                   .pattern())
  {
  }

  state::TimeDims dims() const override
  {
    return {2, 3, 0, 3, 1};
  }

  const HostCsrPattern& hostPattern() const override
  {
    return pattern_;
  }

  const HostCsrPattern& pattern() const override
  {
    return pattern_;
  }

  void initialState(HostConstVectorView prm,
                    HostVector&         out,
                    CpuContext&         ctx) const override
  {
    require(prm.empty(), "Identity residual is parameter-free");
    linalg::HostVectorHandler vec_handler(ctx);
    vec_handler.resizeOrZero(out, 3);
  }

  void res(const state::HostTimeContext& ctx,
           HostVector&                   out,
           CpuContext&) const override
  {
    out = ctx.nxt;
  }

  void applyJacT(const state::HostTimeContext&,
                 state::VariableBlock wrt,
                 HostConstVectorView  adj,
                 HostVector&          out,
                 CpuContext&          ctx) const override
  {
    require(!wrt.isNextState(),
            "Identity transpose apply supports history and parameters");
    if (wrt.isParam())
    {
      out.resize(0);
      return;
    }
    linalg::HostVectorHandler vec_handler(ctx);
    vec_handler.resizeOrZero(out, 3);
  }

  void assembleNext(const state::HostTimeContext& time,
                    HostVector&                   res,
                    HostCsrMatrix&                out,
                    CpuContext&                   ctx) const override
  {
    res = time.nxt;
    linalg::HostMatrixHandler mat_handler(ctx);
    mat_handler.zero(out);
    for (Index i = 0; i < 3; ++i)
    {
      for (Index k = out.rowPtrData()[i]; k < out.rowPtrData()[i + 1]; ++k)
      {
        if (out.colIndData()[k] == i)
        {
          out.valsData()[k] = 1.0;
        }
      }
    }
  }

private:
  HostCsrPattern pattern_;
};

bool near(Real a, Real b)
{
  return std::abs(a - b) <= 1.0e-12;
}

template <std::size_t N>
bool valsNear(const HostVector&          actual,
              const std::array<Real, N>& expected)
{
  if (actual.size() != static_cast<Index>(N))
  {
    return false;
  }
  for (std::size_t i = 0; i < N; ++i)
  {
    if (!near(actual[static_cast<Index>(i)], expected[i]))
    {
      return false;
    }
  }
  return true;
}

fem::DirichletControl mappedControl()
{
  return fem::DirichletControl(
      Array<Index>{0, 2},
      1,
      Array<fem::DirichletControlMapEntry>{{0, 0, 2.0},
                                           {1, 0, -1.0}});
}

TestOutcome mappedTimeDirichletResidual()
{
  TestStatus status(__func__);

  DenseMatrix modes(3, 1);
  modes(1, 0)        = 3.0;
  const auto initial = fem::makeInitialStateMap(
      HostVector{1.0, 2.0, 3.0},
      std::move(modes),
      mappedControl(),
      0,
      0,
      2);
  const IdentityTimeResidual                  base;
  const assembly::HostConstrainedTimeResidual res(
      base,
      fem::makeControlMap(
          2,
          3,
          mappedControl(),
          {},
          {},
          Array<LinearInterpolation>{{0, 1, 0.25}, {1, 1, 0.0}},
          0),
      initial);

  status *= res.dims().num_param == 2;

  const HostVector             history{1.0, 2.0, 3.0};
  const HostVector             next{10.0, 20.0, 30.0};
  const HostVector             parameters{4.0, 8.0};
  const state::HostTimeContext ctx{
      0,
      next.view(),
      parameters.view(),
      state::HostTimeHistoryView(history.data(), 1, 3)};
  CpuContext cpu;

  HostVector out;
  res.initialState(parameters.view(), out, cpu);
  status *= valsNear(out, std::array<Real, 3>{{8.0, 14.0, -4.0}});
  const HostVector initial_adj{1.0, 2.0, 3.0};
  HostVector       initial_grad(2);
  res.addInitialStateJacobianTranspose(
      initial_adj.view(), initial_grad.view(), cpu);
  status *= valsNear(initial_grad, std::array<Real, 2>{{5.0, 0.0}});

  res.res(ctx, out, cpu);
  status *= valsNear(out, std::array<Real, 3>{{0.0, 20.0, 35.0}});

  const HostVector adj{1.0, 5.0, 3.0};
  res.applyJacT(ctx, state::VariableBlock::Param, adj.view(), out, cpu);
  status *= valsNear(out, std::array<Real, 2>{{0.75, 0.25}});

  res.applyJacT(
      ctx, state::VariableBlock::hist(0), adj.view(), out, cpu);
  status *= valsNear(out, std::array<Real, 3>{{0.0, 0.0, 0.0}});

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::mappedTimeDirichletResidual();

  return results.summary();
}
