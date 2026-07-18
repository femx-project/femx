#include <array>
#include <cmath>

#include "TestHelper.hpp"
#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
using namespace fem;

namespace tests
{
namespace
{

class IdentityTimeResidual final : public state::TimeResidual
{
public:
  state::TimeDims dims() const override
  {
    return {2, 3, 0, 3, 1};
  }

  void res(const state::TimeContext& ctx,
           HostVector&               out) const override
  {
    out = *ctx.nxt;
  }

  void applyJac(const state::TimeContext&,
                state::VariableBlock wrt,
                const HostVector&    dir,
                HostVector&          out) const override
  {
    resizeOrZero(out, 3);
    if (wrt.isNextState())
    {
      out = dir;
    }
  }

  void applyJacT(const state::TimeContext&,
                 state::VariableBlock wrt,
                 const HostVector&    adj,
                 HostVector&          out) const override
  {
    if (wrt.isParam())
    {
      out.resize(0);
      return;
    }
    out = adj;
  }

  bool assembleJac(const state::TimeContext&,
                   state::VariableBlock    wrt,
                   linalg::MatrixOperator& out) const override
  {
    const Index cols = wrt.isParam() ? 0 : 3;
    out.resize(3, cols);
    out.setZero();
    if (wrt.isNextState())
    {
      for (Index i = 0; i < 3; ++i)
      {
        out.set(i, i, 1.0);
      }
    }
    return true;
  }
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

  const IdentityTimeResidual                   base;
  const assembly::TimeDirichletControlResidual res(
      base,
      mappedControl(),
      {},
      0,
      -1,
      {},
      Array<LinearInterpolation>{{0, 1, 0.25}, {1, 1, 0.0}});

  status *= res.numParams() == 2;

  const HostVector         history{1.0, 2.0, 3.0};
  const HostVector         next{10.0, 20.0, 30.0};
  const HostVector         parameters{4.0, 8.0};
  const state::TimeContext ctx{
      0,
      &next,
      &parameters,
      state::TimeHistoryView(history.data(), 1, 3)};

  HostVector out;
  res.res(ctx, out);
  status *= valsNear(out, std::array<Real, 3>{{0.0, 20.0, 35.0}});

  res.applyJac(
      ctx, state::VariableBlock::Param, HostVector{2.0, 6.0}, out);
  status *= valsNear(out, std::array<Real, 3>{{-6.0, 0.0, 3.0}});

  res.applyJacT(
      ctx, state::VariableBlock::Param, HostVector{1.0, 5.0, 3.0}, out);
  status *= valsNear(out, std::array<Real, 2>{{0.75, 0.25}});

  DenseMatrix param_jac;
  status *= res.assembleJac(
      ctx, state::VariableBlock::Param, param_jac);
  status *= param_jac.numRows() == 3;
  status *= param_jac.numCols() == 2;
  status *= near(param_jac(0, 0), -1.5);
  status *= near(param_jac(0, 1), -0.5);
  status *= near(param_jac(1, 0), 0.0);
  status *= near(param_jac(1, 1), 0.0);
  status *= near(param_jac(2, 0), 0.75);
  status *= near(param_jac(2, 1), 0.25);

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
