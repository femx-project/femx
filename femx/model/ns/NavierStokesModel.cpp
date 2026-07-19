#include "NavierStokesModel.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/assembly/Assembly.hpp>
#include <femx/common/Checks.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/linalg/Dense.hpp>
#include <femx/model/ns/Helper.hpp>

#if defined(FEMX_HAS_PETSC)
#include <petscsys.h>

#include <femx/linalg/petsc/PETScOperator.hpp>
#endif

namespace femx::model::ns
{
namespace
{

constexpr Index kQuadratureOrder = 2;

void requireModelPrm(Index nstep, Real dt, const FluidParams& fluid)
{
  require(nstep > 0,
          "NavierStokesModel requires a positive number of time steps");
  require(dt > 0.0 && std::isfinite(dt),
          "NavierStokesModel requires a positive finite time step");
  require(std::isfinite(fluid.rho) && fluid.rho > 0.0,
          "NavierStokesModel requires positive finite density");
  require(std::isfinite(fluid.mu) && fluid.mu > 0.0,
          "NavierStokesModel requires positive finite viscosity");
}

fem::Mesh validatedModelMesh(fem::Mesh          mesh,
                             Index              nstep,
                             Real               dt,
                             const FluidParams& fluid)
{
  requireModelPrm(nstep, dt, fluid);
  return mesh;
}

fem::Mesh readModelMesh(const std::string& path,
                        Index              nstep,
                        Real               dt,
                        const FluidParams& fluid)
{
  requireModelPrm(nstep, dt, fluid);
  require(!path.empty(), "NavierStokesModel mesh file is required");
  return fem::GmshReader::read(path);
}

fem::GaussQuadrature makeVelocityQuadrature(
    const fem::MixedFESpace& space)
{
  return fem::GaussQuadrature::make(
      space.field(0).space().finiteElement().referenceElement(),
      kQuadratureOrder);
}

void add(HostVector& vec, Index i, Real val)
{
#pragma omp atomic update
  vec[i] += val;
}

void resetMat(const assembly::HostAssemblyMap& map, HostCsrMatrix& mat)
{
  require(mat.graph().layoutId() == map.graph().layoutId(),
          "Navier Host CSR matrix must use the model AssemblyMap");
  mat.setZero();
}

void addElem(const assembly::HostAssemblyMap& map,
             HostCsrMatrix&                   mat,
             Index                            ie,
             const Array<Index>&,
             const Array<Index>&,
             const DenseMatrix& elem_mat)
{
  assembly::addElem(map, ie, elem_mat, mat, true);
}

#if defined(FEMX_HAS_PETSC)
void resetMat(const assembly::HostAssemblyMap& map,
              linalg::PETScOperator&           mat)
{
  mat.resize(map.numRes(), map.numStates());
  mat.setZero();
}

void addElem(const assembly::HostAssemblyMap&,
             linalg::PETScOperator& mat,
             Index,
             const Array<Index>& rows,
             const Array<Index>& cols,
             const DenseMatrix&  elem_mat)
{
#pragma omp critical(femx_petsc_matrix_set_value)
  {
    mat.addBlock(rows, cols, elem_mat);
  }
}
#endif

#if defined(FEMX_HAS_PETSC)
void allreduce(HostVector& vec)
{
  const int ierr = MPI_Allreduce(MPI_IN_PLACE,
                                 vec.data(),
                                 static_cast<int>(vec.size()),
                                 MPIU_REAL,
                                 MPI_SUM,
                                 PETSC_COMM_WORLD);
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error("Navier residual MPI_Allreduce failed");
  }
}
#endif

} // namespace

template <class Backend>
class HostNavierResidual : public state::TimeResidual<Backend>
{
public:
  using Base      = state::TimeResidual<Backend>;
  using Mat       = typename Base::Mat;
  using Ctx       = typename Base::Ctx;
  using ConstView = typename Base::ConstView;

  HostNavierResidual(Index                             nstep,
                     const assembly::HostAssemblyMap&  map,
                     NavierOperator<MemorySpace::Host> op)
    : nstep_(nstep), map_(map), op_(op), ie_end_(map.numElems())
  {
    static_assert(Backend::space == MemorySpace::Host,
                  "HostNavierResidual requires Host state storage");
  }

  void setElemRange(Index ie_begin, Index ie_end)
  {
    require(ie_begin >= 0 && ie_end >= ie_begin
                && ie_end <= map_.numElems(),
            "Navier residual element range is invalid");
#if !defined(FEMX_HAS_PETSC)
    if (ie_begin != 0 || ie_end != map_.numElems())
    {
      throw std::runtime_error(
          "Navier residual element ranges require PETSc");
    }
#endif
    ie_begin_ = ie_begin;
    ie_end_   = ie_end;
  }

  state::TimeDims dims() const override
  {
    return {nstep_, map_.numStates(), 0, map_.numRes(), kNumHist};
  }

  const HostCsrGraph& hostGraph() const override
  {
    return map_.graph();
  }

  const typename Base::Graph& graph() const override
  {
    return map_.graph();
  }

  void initialState(ConstView prm, HostVector& out, Ctx&) const override
  {
    require(prm.empty(),
            "Navier physics residual is parameter-free");
    resizeOrZero(out, map_.numStates());
  }

  void res(const state::HostTimeContext& ctx,
           HostVector&                   out,
           Ctx&) const override
  {
    checkCtx(ctx);
    resizeOrZero(out, map_.numRes());

#pragma omp parallel
    {
      Work work;
#pragma omp for
      for (Index ie = ie_begin_; ie < ie_end_; ++ie)
      {
        gather(ctx, ie, work);
        const auto  e   = elem(ctx.step, ie, work);
        const auto  map = map_.view();
        const Index nr  = map.numResDofs(ie);
        const Index nc  = map.numStateDofs(ie);
        work.jac.resize(nc);
        for (Index row = 0; row < nr; ++row)
        {
          Real val = 0.0;
          op_.evalRow(e,
                      state::VariableBlock::NextState,
                      row,
                      val,
                      work.jac.view());
          add(out, map.resDof(ie, row), val);
        }
      }
    }
    reduce(out);
  }

  void assemble(const state::HostTimeContext& ctx,
                state::VariableBlock          wrt,
                HostVector&                   res,
                Mat&                          jac,
                Ctx&) const override
  {
    checkCtx(ctx);
    checkWrt(wrt);
    require(!wrt.isParam(),
            "Navier parameter Jacobian is matrix-free");
    resizeOrZero(res, map_.numRes());
    assembleImpl(ctx, wrt, &res, jac);
  }

  void applyJac(const state::HostTimeContext& ctx,
                state::VariableBlock          wrt,
                ConstView                     dir,
                HostVector&                   out,
                Ctx&) const override
  {
    checkCtx(ctx);
    checkWrt(wrt);
    require(dir.size() == (wrt.isParam() ? 0 : map_.numStates()),
            "Navier residual direction size mismatch");

    resizeOrZero(out, map_.numRes());
    if (wrt.isParam())
    {
      return;
    }

#pragma omp parallel
    {
      Work work;
#pragma omp for
      for (Index ie = ie_begin_; ie < ie_end_; ++ie)
      {
        gather(ctx, ie, work);
        const auto  e   = elem(ctx.step, ie, work);
        const auto  map = map_.view();
        const Index nr  = map.numResDofs(ie);
        const Index nc  = map.numStateDofs(ie);
        work.jac.resize(nc);
        for (Index row = 0; row < nr; ++row)
        {
          Real unused = 0.0;
          op_.evalRow(e, wrt, row, unused, work.jac.view());
          Real val = 0.0;
          for (Index col = 0; col < nc; ++col)
          {
            val += work.jac[col] * dir[map.stateDof(ie, col)];
          }
          add(out, map.resDof(ie, row), val);
        }
      }
    }
    reduce(out);
  }

  void applyJacT(const state::HostTimeContext& ctx,
                 state::VariableBlock          wrt,
                 ConstView                     adj,
                 HostVector&                   out,
                 Ctx&) const override
  {
    checkCtx(ctx);
    checkWrt(wrt);
    require(adj.size() == map_.numRes(),
            "Navier residual adjoint size mismatch");
    if (wrt.isParam())
    {
      out.resize(0);
      return;
    }

    resizeOrZero(out, map_.numStates());
#pragma omp parallel
    {
      Work work;
#pragma omp for
      for (Index ie = ie_begin_; ie < ie_end_; ++ie)
      {
        gather(ctx, ie, work);
        const auto  e   = elem(ctx.step, ie, work);
        const auto  map = map_.view();
        const Index nr  = map.numResDofs(ie);
        const Index nc  = map.numStateDofs(ie);
        work.jac.resize(nc);
        for (Index row = 0; row < nr; ++row)
        {
          Real unused = 0.0;
          op_.evalRow(e, wrt, row, unused, work.jac.view());
          const Real val = adj[map.resDof(ie, row)];
          for (Index col = 0; col < nc; ++col)
          {
            add(out,
                map.stateDof(ie, col),
                work.jac[col] * val);
          }
        }
      }
    }
    reduce(out);
  }

  void assembleJac(const state::HostTimeContext& ctx,
                   state::VariableBlock          wrt,
                   Mat&                          out,
                   Ctx&) const override
  {
    checkCtx(ctx);
    checkWrt(wrt);
    require(!wrt.isParam(),
            "Navier parameter Jacobian is matrix-free");
    assembleImpl(ctx, wrt, nullptr, out);
  }

private:
  static constexpr Index kNumHist = 2;

  struct Work
  {
    HostVector   hist;
    HostVector   nxt;
    HostVector   jac;
    DenseMatrix  mat;
    Array<Index> rows;
    Array<Index> cols;
  };

  template <class Matrix>
  void assembleImpl(const state::HostTimeContext& ctx,
                    state::VariableBlock          wrt,
                    HostVector*                   res,
                    Matrix&                       mat) const
  {
    resetMat(map_, mat);

#pragma omp parallel
    {
      Work work;
#pragma omp for
      for (Index ie = ie_begin_; ie < ie_end_; ++ie)
      {
        gather(ctx, ie, work);
        const auto  e     = elem(ctx.step, ie, work);
        const auto  map_v = map_.view();
        const Index nr    = map_v.numResDofs(ie);
        const Index nc    = map_v.numStateDofs(ie);
        work.mat.resize(nr, nc);
        work.rows.resize(nr);
        work.cols.resize(nc);

        for (Index row = 0; row < nr; ++row)
        {
          work.rows[row] = map_v.resDof(ie, row);
          Real val       = 0.0;
          op_.evalRow(e,
                      wrt,
                      row,
                      val,
                      {work.mat.data() + row * nc, nc});
          if (res != nullptr)
          {
            add(*res, work.rows[row], val);
          }
        }
        for (Index col = 0; col < nc; ++col)
        {
          work.cols[col] = map_v.stateDof(ie, col);
        }
        addElem(map_, mat, ie, work.rows, work.cols, work.mat);
      }
    }
    if (res != nullptr)
    {
      reduce(*res);
    }
  }

  void checkCtx(const state::HostTimeContext& ctx) const
  {
    require(ctx.step >= 0 && ctx.step < nstep_,
            "Navier residual step is out of range");
    require(ctx.hist.count() >= kNumHist
                && ctx.hist.stateSize() == map_.numStates()
                && ctx.nxt.size() == map_.numStates() && ctx.prm.empty(),
            "Navier residual vector size mismatch");
  }

  static void checkWrt(state::VariableBlock wrt)
  {
    require(!wrt.isHistoryState()
                || (wrt.historyLag() >= 0 && wrt.historyLag() < kNumHist),
            "Navier residual history lag is out of range");
  }

  void gather(const state::HostTimeContext& ctx,
              Index                         ie,
              Work&                         work) const
  {
    const auto  map = map_.view();
    const Index nc  = map.numStateDofs(ie);
    work.hist.resize(kNumHist * nc);
    work.nxt.resize(nc);
    for (Index lag = 0; lag < kNumHist; ++lag)
    {
      const auto src = ctx.hist.state(lag);
      for (Index col = 0; col < nc; ++col)
      {
        work.hist[lag * nc + col] = src[map.stateDof(ie, col)];
      }
    }
    for (Index col = 0; col < nc; ++col)
    {
      work.nxt[col] = ctx.nxt[map.stateDof(ie, col)];
    }
  }

  static assembly::TimeElementView<MemorySpace::Host> elem(
      Index       step,
      Index       ie,
      const Work& work)
  {
    return {ie, step, kNumHist, work.hist.view(), work.nxt.view()};
  }

  void reduce(HostVector& vec) const
  {
#if defined(FEMX_HAS_PETSC)
    if (ie_begin_ != 0 || ie_end_ != map_.numElems())
    {
      allreduce(vec);
    }
#else
    (void) vec;
#endif
  }

  Index                             nstep_{0};
  const assembly::HostAssemblyMap&  map_;
  NavierOperator<MemorySpace::Host> op_;
  Index                             ie_begin_{0};
  Index                             ie_end_{0};
};

class NavierStokesModel::Residual final
  : public HostNavierResidual<linalg::HostCsrBackend>
{
public:
  using HostNavierResidual::HostNavierResidual;
};

NavierStokesModel::NavierStokesModel(const std::string& path,
                                     Index              nstep,
                                     Real               dt,
                                     FluidParams        fluid)
  : NavierStokesModel(
        readModelMesh(path, nstep, dt, fluid),
        nstep,
        dt,
        fluid)
{
}

NavierStokesModel::NavierStokesModel(fem::Mesh   mesh,
                                     Index       nstep,
                                     Real        dt,
                                     FluidParams fluid)
  : nstep_(nstep),
    dt_(dt),
    mesh_(validatedModelMesh(std::move(mesh), nstep_, dt_, fluid)),
    element_(makeElement(mesh_)),
    space_(makeSpace(mesh_, *element_)),
    geometry_(fem::makeGeometry(mesh_)),
    fluid_(fluid),
    data_(makeNavierData(space_.field(0).space(),
                         makeVelocityQuadrature(space_))),
    map_(assembly::makeAssemblyMap(fem::DofLayout(space_)))
{
  ie_end_ = map_.numElems();
  res_    = std::make_unique<Residual>(nstep_, map_, op());
}

NavierStokesModel::~NavierStokesModel() = default;

Index NavierStokesModel::numSteps() const
{
  return nstep_;
}

Index NavierStokesModel::numStates() const
{
  return space_.numDofs();
}

Real NavierStokesModel::dt() const
{
  return dt_;
}

const FluidParams& NavierStokesModel::fluid() const
{
  return fluid_;
}

const fem::Mesh& NavierStokesModel::mesh() const
{
  return mesh_;
}

const fem::MixedFESpace& NavierStokesModel::space() const
{
  return space_;
}

const fem::HostGeometry& NavierStokesModel::geometry() const
{
  return geometry_;
}

state::HostTimeResidual& NavierStokesModel::residual()
{
  return *res_;
}

const state::HostTimeResidual& NavierStokesModel::residual() const
{
  return *res_;
}

void NavierStokesModel::setElemRange(Index ie_begin, Index ie_end)
{
  ie_begin_ = ie_begin;
  ie_end_   = ie_end;
  res_->setElemRange(ie_begin, ie_end);
}

const assembly::HostAssemblyMap& NavierStokesModel::map() const
{
  return map_;
}

const HostNavierData& NavierStokesModel::data() const
{
  return data_;
}

NavierOperator<MemorySpace::Host> NavierStokesModel::op() const
{
  return {data_.view(), {fluid_.rho, fluid_.mu}, dt_};
}

#if defined(FEMX_HAS_PETSC)
std::unique_ptr<state::TimeResidual<linalg::PetscBackend>>
makePetscTimeResidual(const NavierStokesModel& model)
{
  auto res = std::make_unique<HostNavierResidual<linalg::PetscBackend>>(
      model.numSteps(), model.map(), model.op());
  res->setElemRange(model.ie_begin_, model.ie_end_);
  return res;
}
#endif

Array<Index> NavierStokesModel::velocityDofs() const
{
  const auto  velocity  = space_.field(0);
  const Index num_nodes = mesh_.numNodes();
  const Index num_comps = velocity.numComponents();

  Array<Index> dofs;
  dofs.reserve(num_nodes * num_comps);
  for (Index node = 0; node < num_nodes; ++node)
  {
    for (Index component = 0; component < num_comps; ++component)
    {
      dofs.push_back(velocity.globalDof(node, component));
    }
  }
  return dofs;
}

Array<Index> NavierStokesModel::velocityBoundaryDofs(
    Index boundary_tag) const
{
  return fem::makeVelocityControl(space_, boundary_tag).stateDofs();
}

Array<Index> NavierStokesModel::velocityBoundaryDofs(
    const std::string& boundary_name) const
{
  return fem::makeVelocityControl(space_, boundary_name).stateDofs();
}

} // namespace femx::model::ns
