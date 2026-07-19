#include "NavierStokesModel.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/Assembly.hpp>
#include <femx/assembly/ConstrainedTimeResidual.hpp>
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

struct NavierWork
{
  HostVector   hist;
  HostVector   nxt;
  HostVector   jac;
  HostVector   adj;
  HostVector   vjp;
  DenseMatrix  mat;
  Array<Index> rows;
  Array<Index> cols;
};

template <class Vec, class Ctx>
void resizeAndZero(Vec& out, Index size, Ctx& ctx)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  zero(out.view(), ctx);
}

void gather(const assembly::HostAssemblyMap& map,
            Index                            num_hist,
            HostConstVectorView              hist,
            HostConstVectorView              nxt,
            Index                            ie,
            NavierWork&                      work)
{
  const auto  map_v = map.view();
  const Index nc    = map_v.numStateDofs(ie);
  work.hist.resize(num_hist * nc);
  work.nxt.resize(nc);
  for (Index lag = 0; lag < num_hist; ++lag)
  {
    for (Index col = 0; col < nc; ++col)
    {
      work.hist[lag * nc + col] =
          hist[lag * map.numStates() + map_v.stateDof(ie, col)];
    }
  }
  for (Index col = 0; col < nc; ++col)
  {
    work.nxt[col] = nxt[map_v.stateDof(ie, col)];
  }
}

assembly::TimeElementView<MemorySpace::Host> elem(Index             step,
                                                  Index             num_hist,
                                                  Index             ie,
                                                  const NavierWork& work)
{
  return {ie, step, num_hist, work.hist.view(), work.nxt.view()};
}

void reduce(HostVector& vec,
            Index       ie_begin,
            Index       ie_end,
            Index       num_elems)
{
#if defined(FEMX_HAS_PETSC)
  if (ie_begin != 0 || ie_end != num_elems)
  {
    allreduce(vec);
  }
#else
  (void) vec;
  (void) ie_begin;
  (void) ie_end;
  (void) num_elems;
#endif
}

namespace detail
{

template <class Ctx>
void evalNavierRes(
    const NavierOperator<MemorySpace::Host>& op,
    Index                                    step,
    Index                                    num_hist,
    Index                                    ie_begin,
    Index                                    ie_end,
    const assembly::HostAssemblyMap&         map,
    HostConstVectorView                      hist,
    HostConstVectorView                      nxt,
    HostVector&                              out,
    Ctx&                                     ctx)
{
  resizeAndZero(out, map.numRes(), ctx);
#pragma omp parallel
  {
    NavierWork work;
#pragma omp for
    for (Index ie = ie_begin; ie < ie_end; ++ie)
    {
      gather(map, num_hist, hist, nxt, ie, work);
      const auto  e     = elem(step, num_hist, ie, work);
      const auto  map_v = map.view();
      const Index nr    = map_v.numResDofs(ie);
      const Index nc    = map_v.numStateDofs(ie);
      work.jac.resize(nc);
      for (Index row = 0; row < nr; ++row)
      {
        Real val = 0.0;
        op.evalRow(e,
                   state::VariableBlock::NextState,
                   row,
                   val,
                   work.jac.view());
        add(out, map_v.resDof(ie, row), val);
      }
    }
  }
  reduce(out, ie_begin, ie_end, map.numElems());
}

template <class Matrix, class Ctx>
void assembleNavierNext(
    const NavierOperator<MemorySpace::Host>& op,
    Index                                    step,
    Index                                    num_hist,
    Index                                    ie_begin,
    Index                                    ie_end,
    const assembly::HostAssemblyMap&         map,
    HostConstVectorView                      hist,
    HostConstVectorView                      nxt,
    HostVector&                              res,
    Matrix&                                  mat,
    Ctx&                                     ctx)
{
  resizeAndZero(res, map.numRes(), ctx);
  resetMat(map, mat);
#pragma omp parallel
  {
    NavierWork work;
#pragma omp for
    for (Index ie = ie_begin; ie < ie_end; ++ie)
    {
      gather(map, num_hist, hist, nxt, ie, work);
      const auto  e     = elem(step, num_hist, ie, work);
      const auto  map_v = map.view();
      const Index nr    = map_v.numResDofs(ie);
      const Index nc    = map_v.numStateDofs(ie);
      work.mat.resize(nr, nc);
      work.rows.resize(nr);
      work.cols.resize(nc);

      for (Index row = 0; row < nr; ++row)
      {
        work.rows[row] = map_v.resDof(ie, row);
        Real val       = 0.0;
        op.evalRow(e,
                   state::VariableBlock::NextState,
                   row,
                   val,
                   {work.mat.data() + row * nc, nc});
        add(res, work.rows[row], val);
      }
      for (Index col = 0; col < nc; ++col)
      {
        work.cols[col] = map_v.stateDof(ie, col);
      }
      addElem(map, mat, ie, work.rows, work.cols, work.mat);
    }
  }
  reduce(res, ie_begin, ie_end, map.numElems());
}

template <class Ctx>
void applyNavierHistJacT(
    const NavierOperator<MemorySpace::Host>& op,
    Index                                    step,
    Index                                    num_hist,
    Index                                    lag,
    Index                                    ie_begin,
    Index                                    ie_end,
    const assembly::HostAssemblyMap&         map,
    HostConstVectorView                      hist,
    HostConstVectorView                      nxt,
    HostConstVectorView                      adj,
    HostVector&                              out,
    Ctx&                                     ctx)
{
  resizeAndZero(out, map.numStates(), ctx);
#pragma omp parallel
  {
    NavierWork work;
#pragma omp for
    for (Index ie = ie_begin; ie < ie_end; ++ie)
    {
      gather(map, num_hist, hist, nxt, ie, work);
      const auto  e     = elem(step, num_hist, ie, work);
      const auto  map_v = map.view();
      const Index nr    = map_v.numResDofs(ie);
      const Index nc    = map_v.numStateDofs(ie);
      work.adj.resize(nr);
      work.vjp.resize(num_hist * nc);
      for (Index row = 0; row < nr; ++row)
      {
        work.adj[row] = adj[map_v.resDof(ie, row)];
      }
      histVjp(op, e, work.adj.view(), work.vjp.view());
      for (Index col = 0; col < nc; ++col)
      {
        add(out,
            map_v.stateDof(ie, col),
            work.vjp[lag * nc + col]);
      }
    }
  }
  reduce(out, ie_begin, ie_end, map.numElems());
}

} // namespace detail

template <class Backend>
class NavierResidual : public state::TimeResidual<Backend>
{
public:
  using Base      = state::TimeResidual<Backend>;
  using Vec       = typename Base::Vec;
  using ConstView = typename Base::ConstView;
  using Mat       = typename Base::Mat;
  using Graph     = typename Base::Graph;
  using Ctx       = typename Base::Ctx;
  using StepCtx   = typename Base::StepCtx;
  using Map       = assembly::AssemblyMap<Backend::space>;
  using Data      = NavierData<Backend::space>;
  using Op        = NavierOperator<Backend::space>;

  NavierResidual(Index                             nstep,
                 const assembly::HostAssemblyMap&  map,
                 NavierOperator<MemorySpace::Host> op)
    : nstep_(nstep)
  {
    if constexpr (Backend::space == MemorySpace::Host)
    {
      map_ptr_    = &map;
      host_graph_ = &map.graph();
      op_         = op;
      ie_end_     = map.numElems();
    }
    else
    {
      require(false, "Host Navier residual requires Host storage");
    }
  }

  NavierResidual(Index                            nstep,
                 const assembly::HostAssemblyMap& map,
                 const HostNavierData&            data,
                 KernelFluid                      fluid,
                 Real                             dt,
                 Ctx&                             ctx)
    : nstep_(nstep)
  {
    if constexpr (Backend::space == MemorySpace::Device)
    {
      owned_map_ = std::make_unique<Map>();
      copy(map, *owned_map_, ctx);
      owned_data_ = std::make_unique<Data>();
      ns::copy(data, *owned_data_, ctx);
      host_graph_store_ = map.graph();
      map_ptr_          = owned_map_.get();
      host_graph_       = &host_graph_store_;
      op_               = Op(owned_data_->view(), fluid, dt);
      ie_end_           = map.numElems();
    }
    else
    {
      require(false, "Device Navier residual requires Device storage");
    }
  }

  void setElemRange(Index ie_begin, Index ie_end)
  {
    require(ie_begin >= 0 && ie_end >= ie_begin
                && ie_end <= map().numElems(),
            "Navier residual element range is invalid");
    if constexpr (Backend::space == MemorySpace::Device)
    {
      require(ie_begin == 0 && ie_end == map().numElems(),
              "CUDA Navier residual requires the full element range");
    }
#if !defined(FEMX_HAS_PETSC)
    else
    {
      require(ie_begin == 0 && ie_end == map().numElems(),
              "Navier residual element ranges require PETSc");
    }
#endif
    ie_begin_ = ie_begin;
    ie_end_   = ie_end;
  }

  state::TimeDims dims() const override
  {
    return {nstep_, map().numStates(), 0, map().numRes(), kNumHist};
  }

  const HostCsrGraph& hostGraph() const override
  {
    return *host_graph_;
  }

  const Graph& graph() const override
  {
    return map().graph();
  }

  void initialState(ConstView prm, Vec& out, Ctx& ctx) const override
  {
    require(prm.empty(), "Navier physics residual is parameter-free");
    resizeAndZero(out, map().numStates(), ctx);
  }

  void res(const StepCtx& time, Vec& out, Ctx& ctx) const override
  {
    checkCtx(time);
    const ConstView hist{time.hist.data(), kNumHist * map().numStates()};
    detail::evalNavierRes(op_,
                          time.step,
                          kNumHist,
                          ie_begin_,
                          ie_end_,
                          map(),
                          hist,
                          time.nxt,
                          out,
                          ctx);
  }

  void assembleNext(const StepCtx& time,
                    Vec&           res,
                    Mat&           jac,
                    Ctx&           ctx) const override
  {
    checkCtx(time);
    const ConstView hist{time.hist.data(), kNumHist * map().numStates()};
    detail::assembleNavierNext(op_,
                               time.step,
                               kNumHist,
                               ie_begin_,
                               ie_end_,
                               map(),
                               hist,
                               time.nxt,
                               res,
                               jac,
                               ctx);
  }

  void applyJacT(const StepCtx&       time,
                 state::VariableBlock wrt,
                 ConstView            adj,
                 Vec&                 out,
                 Ctx&                 ctx) const override
  {
    checkCtx(time);
    require(!wrt.isNextState(),
            "Navier transpose apply supports only history and parameter blocks");
    require(adj.size() == map().numRes(),
            "Navier residual adjoint size mismatch");
    if (wrt.isParam())
    {
      out.resize(0);
      return;
    }
    require(wrt.historyLag() >= 0 && wrt.historyLag() < kNumHist,
            "Navier residual history lag is out of range");
    if (!ad::has_enzyme)
    {
      throw std::runtime_error(
          "Navier history VJP requires Enzyme. Configure with "
          "-DFEMX_ENABLE_ENZYME=ON and provide Enzyme_DIR.");
    }

    const ConstView hist{time.hist.data(), kNumHist * map().numStates()};
    detail::applyNavierHistJacT(op_,
                                time.step,
                                kNumHist,
                                wrt.historyLag(),
                                ie_begin_,
                                ie_end_,
                                map(),
                                hist,
                                time.nxt,
                                adj,
                                out,
                                ctx);
  }

private:
  const Map& map() const
  {
    return *map_ptr_;
  }

  void checkCtx(const StepCtx& ctx) const
  {
    require(ctx.step >= 0 && ctx.step < nstep_,
            "Navier residual step is out of range");
    require(ctx.hist.count() >= kNumHist
                && ctx.hist.stateSize() == map().numStates()
                && ctx.nxt.size() == map().numStates() && ctx.prm.empty(),
            "Navier residual vector size mismatch");
  }

  Index                 nstep_{0};
  std::unique_ptr<Map>  owned_map_;
  std::unique_ptr<Data> owned_data_;
  const Map*            map_ptr_{nullptr};
  HostCsrGraph          host_graph_store_;
  const HostCsrGraph*   host_graph_{nullptr};
  Op                    op_;
  Index                 ie_begin_{0};
  Index                 ie_end_{0};
};

class NavierStokesModel::Residual final
  : public NavierResidual<linalg::HostCsrBackend>
{
public:
  using NavierResidual::NavierResidual;
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

#if defined(FEMX_HAS_CUDA)
std::unique_ptr<state::DeviceTimeResidual> makeDeviceTimeResidual(
    const NavierStokesModel& model,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init_state)
{
  CudaContext ctx;
  auto        base = std::make_unique<NavierResidual<linalg::CudaCsrBackend>>(
      model.numSteps(),
      model.map(),
      model.data(),
      KernelFluid{model.fluid().rho, model.fluid().mu},
      model.dt(),
      ctx);
  auto out = std::make_unique<assembly::DeviceConstrainedTimeResidual>(
      std::move(base), std::move(control), std::move(init_state), ctx);
  ctx.synchronize();
  return out;
}
#endif

#if defined(FEMX_HAS_PETSC)
std::unique_ptr<state::TimeResidual<linalg::PetscBackend>>
makePetscTimeResidual(const NavierStokesModel& model)
{
  auto res = std::make_unique<NavierResidual<linalg::PetscBackend>>(
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
