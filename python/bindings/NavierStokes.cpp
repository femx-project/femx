#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include "Bindings.hpp"
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/common/LinearInterpolation.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/TimeDirichletData.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/FluidProperties.hpp>
#include <femx/model/ns/Model.hpp>
#include <femx/model/ns/StateFields.hpp>
#ifdef FEMX_RESOLVE_USE_CUDA
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#endif
#include <femx/runtime/LinearSystemFactory.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeResidual.hpp>
#include <femx/state/TimeTrajectory.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{

using femx::DenseMatrix;
using femx::HostVector;
using femx::Index;
using femx::Real;
using femx::assembly::HostConstrainedTimeResidual;
using femx::model::ns::FluidProperties;
using femx::model::ns::NavierStokesModel;
using TimeResidual = femx::state::HostTimeResidual;
using femx::state::TimeTrajectory;

using RealArray  = py::array_t<Real,
                               py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<Index,
                               py::array::c_style | py::array::forcecast>;

py::array_t<Index> indexArray(const HostVector<Index>& vals)
{
  py::array_t<Index> out(vals.size());
  auto               data = out.mutable_unchecked<1>();
  for (Index i = 0; i < vals.size(); ++i)
  {
    data(i) = vals[i];
  }
  return out;
}

HostVector<Real> realVector(const RealArray& vals, const char* name)
{
  if (vals.ndim() != 1)
  {
    throw std::runtime_error(std::string(name) + " must be one-dimensional");
  }
  HostVector<Real> out(vals.shape(0));
  const auto       data = vals.unchecked<1>();
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = data(i);
    if (!std::isfinite(out[i]))
    {
      throw std::runtime_error(std::string(name) + " must be finite");
    }
  }
  return out;
}

py::array_t<Real> vectorArray(const HostVector<Real>& vals)
{
  py::array_t<Real> out(vals.size());
  auto              data = out.mutable_unchecked<1>();
  for (Index i = 0; i < vals.size(); ++i)
  {
    data(i) = vals[i];
  }
  return out;
}

DenseMatrix realMatrix(const RealArray& vals, const char* name)
{
  if (vals.ndim() != 2)
  {
    throw std::runtime_error(std::string(name) + " must be two-dimensional");
  }
  DenseMatrix out(vals.shape(0), vals.shape(1));
  const auto  data = vals.unchecked<2>();
  for (Index row = 0; row < out.rows(); ++row)
  {
    for (Index col = 0; col < out.cols(); ++col)
    {
      out(row, col) = data(row, col);
      if (!std::isfinite(out(row, col)))
      {
        throw std::runtime_error(std::string(name) + " must be finite");
      }
    }
  }
  return out;
}

py::array_t<Index> boundaryVelocityDofs(
    const NavierStokesModel& model,
    const py::object&        selector)
{
  if (py::isinstance<py::str>(selector))
  {
    return indexArray(
        model.velocityBoundaryDofs(selector.cast<std::string>()));
  }
  if (py::isinstance<py::int_>(selector)
      && !py::isinstance<py::bool_>(selector))
  {
    return indexArray(model.velocityBoundaryDofs(selector.cast<Index>()));
  }
  throw std::runtime_error("Boundary selector must be a physical name or tag");
}

void writeNavierStokesXdmf(const NavierStokesModel& model,
                           const std::string&       path,
                           const TimeTrajectory&    trajectory)
{
  if (path.empty())
  {
    throw std::runtime_error("XDMF output path must not be empty");
  }
  if (trajectory.numSteps() != model.numSteps()
      || trajectory.numStates() != model.numStates())
  {
    throw std::runtime_error(
        "XDMF trajectory dimensions do not match the Navier-Stokes model");
  }

  const Index      num_nodes = model.mesh().numNodes();
  HostVector<Real> ux(num_nodes);
  HostVector<Real> uy(num_nodes);
  HostVector<Real> uz(num_nodes);
  HostVector<Real> pressure(num_nodes);

  femx::io::TimeSeriesDataOut out;
  out.attachMesh(model.mesh());
  for (Index level = 0; level < trajectory.numTimeLevels(); ++level)
  {
    femx::model::ns::splitStateFields(trajectory[level],
                                      model.space(),
                                      ux,
                                      uy,
                                      uz,
                                      pressure);
    out.beginStep(static_cast<Real>(level) * model.dt());
    out.addNodalVectorField("velocity", ux, uy, uz);
    out.addNodalScalarField("pressure", pressure);
  }
  out.write(path);
}

Index fieldId(const std::string& field)
{
  if (field == "velocity")
  {
    return 0;
  }
  if (field == "pressure")
  {
    return 1;
  }
  throw std::runtime_error("Unknown Navier-Stokes field: " + field);
}

template <typename Match>
std::set<Index> boundaryNodes(const NavierStokesModel& model, Match match)
{
  const auto&     mesh = model.mesh();
  std::set<Index> nodes;
  for (const auto& facet : mesh.boundaryFacets())
  {
    if (match(facet))
    {
      nodes.insert(facet.nids.begin(), facet.nids.end());
    }
  }
  if (nodes.empty())
  {
    throw std::runtime_error("No boundary facets matched the selector");
  }

  return nodes;
}

std::set<Index> selectBoundaryNodes(const NavierStokesModel& model,
                                    const py::object&        selector)
{
  if (py::isinstance<py::str>(selector))
  {
    const std::string name = selector.cast<std::string>();
    return boundaryNodes(
        model,
        [&name](const femx::fem::Mesh::BoundaryFacet& facet)
        {
          return facet.pname == name;
        });
  }
  if (py::isinstance<py::int_>(selector)
      && !py::isinstance<py::bool_>(selector))
  {
    const Index tag = selector.cast<Index>();
    return boundaryNodes(
        model,
        [tag](const femx::fem::Mesh::BoundaryFacet& facet)
        {
          return facet.ptag == tag;
        });
  }
  throw std::runtime_error("Boundary selector must be a physical name or tag");
}

py::array_t<Real> pointArray(const femx::fem::Mesh::Node& point, Index dim)
{
  py::array_t<Real> out(dim);
  auto              vals = out.mutable_unchecked<1>();
  for (Index component = 0; component < dim; ++component)
  {
    vals(component) = point[component];
  }
  return out;
}

HostVector<Real> pythonBoundaryValue(const py::object&            value,
                                     const femx::fem::Mesh::Node& point,
                                     Index                        dim,
                                     Real                         time,
                                     Index                        components,
                                     const std::string&           field)
{
  const py::object raw  = PyCallable_Check(value.ptr())
                              ? value(pointArray(point, dim), time)
                              : value;
  const RealArray  vals = RealArray::ensure(raw);
  if (!vals)
  {
    throw std::runtime_error(field + " boundary value must be real-valued");
  }

  HostVector<Real> out(components);
  if (components == 1 && vals.ndim() == 0)
  {
    out[0] = *vals.data();
  }
  else if (vals.ndim() == 1 && vals.shape(0) == components)
  {
    const auto data = vals.unchecked<1>();
    for (Index component = 0; component < components; ++component)
    {
      out[component] = data(component);
    }
  }
  else
  {
    const std::string expected = components == 1
                                     ? "a scalar"
                                     : std::to_string(components) + " values";
    throw std::runtime_error(
        field + " boundary value must return " + expected);
  }
  for (Real component : out)
  {
    if (!std::isfinite(component))
    {
      throw std::runtime_error(field + " boundary value must be finite");
    }
  }
  return out;
}

femx::fem::DirichletBC pythonDirichletBC(
    const NavierStokesModel& model,
    const py::list&          specifications,
    Real                     time)
{
  femx::fem::DirichletBC out;
  bool                   has_pressure = false;
  for (const py::handle item : specifications)
  {
    const py::object specification =
        py::reinterpret_borrow<py::object>(item);
    const std::string field =
        specification.attr("field").cast<std::string>();
    const py::object selector    = specification.attr("boundary");
    const py::object value       = specification.attr("value");
    const auto       field_view  = model.space().field(fieldId(field));
    has_pressure                |= field == "pressure";

    for (Index node : selectBoundaryNodes(model, selector))
    {
      const HostVector<Real> components = pythonBoundaryValue(
          value,
          model.mesh().node(node),
          model.mesh().dim(),
          time,
          field_view.numComponents(),
          field);
      for (Index component = 0; component < components.size(); ++component)
      {
        out.addDof(field_view.globalDof(node, component),
                   components[component]);
      }
    }
  }

  if (!has_pressure)
  {
    out.addDof(model.space().field(fieldId("pressure")).globalDof(0), 0.0);
  }
  return out;
}

femx::fem::TimeDirichletData makePythonDirichletData(
    const NavierStokesModel& model,
    const py::iterable&      specifications)
{
  py::list items;
  for (const py::handle item : specifications)
  {
    items.append(item);
  }
  return femx::fem::makeTimeDirichletData(
      model.numStates(),
      model.numSteps(),
      model.dt(),
      [&model, &items](Real time)
      {
        return pythonDirichletBC(model, items, time);
      });
}

HostVector<Real> pythonNormal(const NavierStokesModel& model,
                              const py::object&        specification)
{
  const RealArray vals =
      RealArray::ensure(specification.attr("normal"));
  const Index components = model.space().field(fieldId("velocity")).numComponents();
  if (!vals || vals.ndim() != 1 || vals.shape(0) != components)
  {
    throw std::runtime_error(
        "normal must contain one value per velocity component");
  }

  HostVector<Real> normal(components);
  const auto       data = vals.unchecked<1>();
  for (Index component = 0; component < components; ++component)
  {
    normal[component] = data(component);
  }
  return normal;
}

femx::fem::DirichletControl makePythonNormalControl(
    const NavierStokesModel& model,
    const py::object&        specification,
    const HostVector<Index>& fixed_dofs)
{
  const py::object            selector = specification.attr("boundary");
  const HostVector<Real>      normal   = pythonNormal(model, specification);
  femx::fem::DirichletControl control;
  if (py::isinstance<py::str>(selector))
  {
    control = femx::fem::makeNormalVelocityControl(
        model.space(), selector.cast<std::string>(), normal);
  }
  else if (py::isinstance<py::int_>(selector)
           && !py::isinstance<py::bool_>(selector))
  {
    control = femx::fem::makeNormalVelocityControl(
        model.space(), selector.cast<Index>(), normal);
  }
  else
  {
    throw std::runtime_error(
        "Boundary selector must be a physical name or tag");
  }

  control = control.withoutStateDofs(fixed_dofs);
  if (control.numStateDofs() == 0 || control.numControlParams() == 0)
  {
    throw std::runtime_error(
        "Normal velocity control has no active boundary nodes");
  }
  return control;
}

femx::fem::DirichletControl makePythonVelocityControl(
    const NavierStokesModel& model,
    const py::object&        specification,
    const HostVector<Index>& fixed_dofs)
{
  const py::object            selector = specification.attr("boundary");
  femx::fem::DirichletControl control;
  if (py::isinstance<py::str>(selector))
  {
    control = femx::fem::makeVelocityControl(
        model.space(), selector.cast<std::string>());
  }
  else if (py::isinstance<py::int_>(selector)
           && !py::isinstance<py::bool_>(selector))
  {
    control = femx::fem::makeVelocityControl(
        model.space(), selector.cast<Index>());
  }
  else
  {
    throw std::runtime_error("Boundary selector must be a physical name or tag");
  }
  control = control.withoutStateDofs(fixed_dofs);
  if (control.numStateDofs() == 0 || control.numControlParams() == 0)
  {
    throw std::runtime_error("Velocity control has no active boundary dofs");
  }
  return control;
}

femx::fem::DirichletControl makePythonControl(
    const NavierStokesModel& model,
    const py::object&        specification,
    const HostVector<Index>& fixed_dofs)
{
  const std::string kind = specification.attr("kind").cast<std::string>();
  if (kind == "normal")
  {
    return makePythonNormalControl(model, specification, fixed_dofs);
  }
  if (kind == "vector")
  {
    return makePythonVelocityControl(model, specification, fixed_dofs);
  }
  throw std::runtime_error("control kind must be 'normal' or 'vector'");
}

HostVector<femx::LinearInterpolation> makePythonControlTimeStencils(
    const NavierStokesModel& model,
    const py::object&        specification)
{
  const py::object raw_times = specification.attr("times");
  if (raw_times.is_none())
  {
    return {};
  }

  const RealArray vals = RealArray::ensure(raw_times);
  if (!vals || vals.ndim() != 1 || vals.shape(0) == 0)
  {
    throw std::runtime_error(
        "control times must be a nonempty one-dimensional array");
  }

  HostVector<Real> times(vals.shape(0));
  const auto       data = vals.unchecked<1>();
  for (Index level = 0; level < times.size(); ++level)
  {
    times[level] = data(level);
    if (!std::isfinite(times[level]) || times[level] < 0.0
        || (level > 0 && times[level] <= times[level - 1]))
    {
      throw std::runtime_error(
          "control times must be finite, nonnegative, and increasing");
    }
  }
  const Real final_time = model.numSteps() * model.dt();
  if (times.back() > final_time)
  {
    throw std::runtime_error(
        "control times must not extend past the final solve time");
  }

  const bool periodic = specification.attr("periodic").cast<bool>();
  if (periodic
      && (times.size() < 2 || times.front() != 0.0
          || std::abs(times.back() - final_time)
                 > 1.0e-12 * std::max(1.0, std::abs(final_time))))
  {
    throw std::runtime_error(
        "periodic control times must start at zero and end at the final solve time");
  }

  HostVector<femx::LinearInterpolation> stencils(model.numSteps());
  for (Index step = 0; step < model.numSteps(); ++step)
  {
    const Real time = static_cast<Real>(step + 1) * model.dt();
    stencils[step]  = femx::linearInterpolation(times, time);
    if (periodic)
    {
      const Index last = times.size() - 1;
      if (stencils[step].lower == last)
      {
        stencils[step] = {0, 0, 0.0};
      }
      else if (stencils[step].upper == last)
      {
        stencils[step] =
            stencils[step].lower == 0
                ? femx::LinearInterpolation{0, 0, 0.0}
                : femx::LinearInterpolation{
                      0,
                      stencils[step].lower,
                      stencils[step].lowerWeight()};
      }
    }
  }

  const Index     num_levels = periodic ? times.size() - 1 : times.size();
  std::set<Index> used_levels;
  for (const femx::LinearInterpolation& stencil : stencils)
  {
    if (stencil.lowerWeight() != 0.0)
    {
      used_levels.insert(stencil.lower);
    }
    if (stencil.hasUpper())
    {
      used_levels.insert(stencil.upper);
    }
  }
  for (Index level = 0; level < num_levels; ++level)
  {
    if (used_levels.find(level) == used_levels.end())
    {
      throw std::runtime_error(
          "control time level at t=" + std::to_string(times[level])
          + " is not sampled by any solver step; refine the solver time step "
            "or remove the unused control time");
    }
  }
  return stencils;
}

HostVector<femx::Point3> pythonObservationPoints(
    const NavierStokesModel& model,
    const RealArray&         vals)
{
  const Index dim = model.mesh().dim();
  if (vals.ndim() != 2 || vals.shape(0) == 0
      || (vals.shape(1) != dim && vals.shape(1) != 3))
  {
    throw std::runtime_error(
        "observation points must have shape (num_points, mesh_dimension) or "
        "(num_points, 3)");
  }

  HostVector<femx::Point3> points(vals.shape(0));
  const auto               data = vals.unchecked<2>();
  for (Index point = 0; point < points.size(); ++point)
  {
    points[point] = {0.0, 0.0, 0.0};
    for (Index axis = 0; axis < vals.shape(1); ++axis)
    {
      const Real value = data(point, axis);
      if (!std::isfinite(value))
      {
        throw std::runtime_error(
            "observation points must be finite");
      }
      points[point][axis] = value;
    }
  }
  return points;
}

HostVector<Index> pythonObservationComponents(
    const NavierStokesModel& model,
    const IndexArray&        vals)
{
  if (vals.ndim() != 1 || vals.shape(0) == 0)
  {
    throw std::runtime_error(
        "observation components must be a nonempty one-dimensional array");
  }

  const Index       num_components = model.space().field(0).numComponents();
  HostVector<Index> components(vals.shape(0));
  std::set<Index>   seen;
  const auto        data = vals.unchecked<1>();
  for (Index i = 0; i < components.size(); ++i)
  {
    components[i] = data(i);
    if (components[i] < 0 || components[i] >= num_components)
    {
      throw std::runtime_error(
          "observation component is out of range");
    }
    if (!seen.insert(components[i]).second)
    {
      throw std::runtime_error(
          "observation components must not contain duplicates");
    }
  }
  return components;
}

py::array_t<Real> timeDirichletValueArray(
    const femx::fem::TimeDirichletData& data,
    Index                               steps)
{
  py::array_t<Real> out({steps, data.dofs.size()});
  auto              vals = out.mutable_unchecked<2>();
  for (Index step = 0; step < steps; ++step)
  {
    for (Index col = 0; col < data.dofs.size(); ++col)
    {
      vals(step, col) = data.vals[step * data.dofs.size() + col];
    }
  }
  return out;
}

class FixedDirichletProblem
{
public:
  FixedDirichletProblem(NavierStokesModel&  model,
                        const py::iterable& specifications)
    : model_(model),
      data_(makePythonDirichletData(model, specifications)),
      res_(model_.residual(),
           femx::fem::makeControlMap(model_.numSteps(),
                                     model_.numStates(),
                                     {},
                                     data_.dofs,
                                     data_.vals,
                                     {},
                                     0,
                                     0))
  {
  }

  TimeResidual& residual()
  {
    return res_;
  }

  const femx::fem::TimeDirichletData& data() const
  {
    return data_;
  }

  const femx::fem::HostControlMap& controlMap() const
  {
    return res_.controlMap();
  }

  Index numSteps() const
  {
    return model_.numSteps();
  }

private:
  NavierStokesModel&           model_;
  femx::fem::TimeDirichletData data_;
  HostConstrainedTimeResidual  res_;
};

class ControlledDirichletProblem
{
public:
  ControlledDirichletProblem(NavierStokesModel&  model,
                             const py::iterable& boundary_conditions,
                             const py::object&   ctr_specification,
                             Index               ctr_param_offset)
    : model_(model),
      data_(makePythonDirichletData(model, boundary_conditions)),
      ctr_(makePythonControl(
          model, ctr_specification, data_.dofs)),
      time_stencils_(makePythonControlTimeStencils(
          model, ctr_specification)),
      res_(model_.residual(),
           femx::fem::makeControlMap(model_.numSteps(),
                                     model_.numStates(),
                                     ctr_,
                                     data_.dofs,
                                     data_.vals,
                                     time_stencils_,
                                     ctr_param_offset)),
      ctr_param_offset_(ctr_param_offset)
  {
  }

  TimeResidual& residual()
  {
    return res_;
  }

  const femx::fem::TimeDirichletData& data() const
  {
    return data_;
  }

  const femx::fem::DirichletControl& control() const
  {
    return ctr_;
  }

  Index numSteps() const
  {
    return model_.numSteps();
  }

  Index numControlLevels() const
  {
    return (res_.dims().num_param - ctr_param_offset_)
           / ctr_.numControlParams();
  }

  Index controlParamOffset() const
  {
    return ctr_param_offset_;
  }

  const HostVector<femx::LinearInterpolation>& timeStencils() const
  {
    return time_stencils_;
  }

  const femx::fem::HostControlMap& controlMap() const
  {
    return res_.controlMap();
  }

  HostVector<Index> controlMeshNodeIds() const
  {
    const auto                 u_dof = model_.space().field(0);
    const Index                comps = u_dof.numComponents();
    HostVector<Index>          nodes(ctr_.numControlParams(), -1);
    const femx::HostCsrMatrix& matrix = ctr_.matrix();
    for (Index row = 0; row < matrix.rows(); ++row)
    {
      const Index dof  = ctr_.stateDof(row);
      const Index node = dof / comps;
      const Index comp = dof % comps;
      if (node < 0 || node >= model_.mesh().numNodes()
          || u_dof.globalDof(node, comp) != dof)
      {
        throw std::runtime_error(
            "Normal velocity control contains a non-velocity dof");
      }
      for (Index k = matrix.rowPtrData()[row];
           k < matrix.rowPtrData()[row + 1];
           ++k)
      {
        Index& stored = nodes[matrix.colIndData()[k]];
        if (stored >= 0 && stored != node)
        {
          throw std::runtime_error(
              "Normal velocity control column spans multiple nodes");
        }
        stored = node;
      }
    }
    for (const Index node : nodes)
    {
      if (node < 0)
      {
        throw std::runtime_error(
            "Normal velocity control contains an unused column");
      }
    }
    return nodes;
  }

  femx::fem::HostInitialStateMap makeInitialStateMap(
      const RealArray& mean,
      const RealArray& modes)
  {
    auto map = femx::fem::makeInitialStateMap(
        realVector(mean, "initial_state_mean"),
        realMatrix(modes, "initial_state_modes"),
        ctr_,
        0,
        ctr_param_offset_,
        res_.dims().num_param);
    res_.setInitialStateMap(map);
    return map;
  }

private:
  NavierStokesModel&                    model_;
  femx::fem::TimeDirichletData          data_;
  femx::fem::DirichletControl           ctr_;
  HostVector<femx::LinearInterpolation> time_stencils_;
  HostConstrainedTimeResidual           res_;
  Index                                 ctr_param_offset_{0};
};

class VelocityPointSampler final
  : public femx::inverse::TimeObservationOperator
{
public:
  VelocityPointSampler(const NavierStokesModel& model,
                       const RealArray&         points,
                       const IndexArray&        components,
                       Index                    num_param)
    : interpolator_(model.numSteps(),
                    model.space(),
                    0,
                    pythonObservationPoints(model, points),
                    pythonObservationComponents(model, components),
                    num_param)
  {
  }

  Index numSteps() const override
  {
    return interpolator_.numSteps();
  }

  Index numStates() const override
  {
    return interpolator_.numStates();
  }

  Index numParams() const override
  {
    return interpolator_.numParams();
  }

  Index numObservations() const override
  {
    return interpolator_.numObservations();
  }

  std::unique_ptr<femx::inverse::DeviceTimeObservationOperator>
  copyToDevice(femx::linalg::CudaContext& ctx) const override
  {
    return interpolator_.copyToDevice(ctx);
  }

  void observe(Index                   level,
               const HostVector<Real>& state,
               const HostVector<Real>& prm,
               HostVector<Real>&       out) const override
  {
    interpolator_.observe(level, state, prm, out);
  }

  void applyStateJac(Index                   level,
                     const HostVector<Real>& state,
                     const HostVector<Real>& prm,
                     const HostVector<Real>& dir,
                     HostVector<Real>&       out) const override
  {
    interpolator_.applyStateJac(level, state, prm, dir, out);
  }

  void applyStateJacT(Index                   level,
                      const HostVector<Real>& state,
                      const HostVector<Real>& prm,
                      const HostVector<Real>& dir,
                      HostVector<Real>&       out) const override
  {
    interpolator_.applyStateJacT(level, state, prm, dir, out);
  }

  void applyParamJac(Index                   level,
                     const HostVector<Real>& state,
                     const HostVector<Real>& prm,
                     const HostVector<Real>& dir,
                     HostVector<Real>&       out) const override
  {
    interpolator_.applyParamJac(level, state, prm, dir, out);
  }

  void applyParamJacT(Index                   level,
                      const HostVector<Real>& state,
                      const HostVector<Real>& prm,
                      const HostVector<Real>& dir,
                      HostVector<Real>&       out) const override
  {
    interpolator_.applyParamJacT(level, state, prm, dir, out);
  }

  py::array_t<Real> sample(const femx::state::TimeTrajectory& trajectory) const
  {
    if (trajectory.numSteps() != interpolator_.numSteps()
        || trajectory.numStates() != interpolator_.numStates())
    {
      throw std::runtime_error(
          "observation trajectory dimensions do not match the model");
    }

    const Index       num_points     = interpolator_.pts().size();
    const Index       num_components = interpolator_.comps().size();
    py::array_t<Real> out(
        {trajectory.numTimeLevels(), num_points, num_components});
    auto             data = out.mutable_unchecked<3>();
    HostVector<Real> parameters(numParams());
    for (Index level = 0; level < trajectory.numTimeLevels(); ++level)
    {
      HostVector<Real> vals;
      observe(level, trajectory[level], parameters, vals);
      for (Index point = 0; point < num_points; ++point)
      {
        for (Index component = 0; component < num_components; ++component)
        {
          data(level, point, component) =
              vals[point * num_components + component];
        }
      }
    }
    return out;
  }

  py::array_t<Real> applyTranspose(const RealArray& directions) const
  {
    if (directions.ndim() != 2
        || directions.shape(0) != numSteps() + 1
        || directions.shape(1) != numObservations())
    {
      throw std::runtime_error(
          "solve-level observation directions have inconsistent dimensions");
    }

    py::array_t<Real> out({numSteps() + 1, numStates()});
    auto              output = out.mutable_unchecked<2>();
    const auto        input  = directions.unchecked<2>();
    HostVector<Real>  state(numStates());
    HostVector<Real>  parameters(numParams());
    for (Index level = 0; level <= numSteps(); ++level)
    {
      HostVector<Real> direction(numObservations());
      for (Index i = 0; i < numObservations(); ++i)
      {
        direction[i] = input(level, i);
        if (!std::isfinite(direction[i]))
        {
          throw std::runtime_error(
              "observation directions must be finite");
        }
      }

      HostVector<Real> gradient;
      applyStateJacT(level, state, parameters, direction, gradient);
      for (Index i = 0; i < numStates(); ++i)
      {
        output(level, i) = gradient[i];
      }
    }
    return out;
  }

private:
  femx::fem::TimePointInterpolator interpolator_;
};

#ifdef FEMX_RESOLVE_USE_CUDA
femx::linalg::ReSolveOptions resolveOptions(const py::object& obj)
{
  return obj.is_none()
             ? femx::linalg::ReSolveOptions{}
             : obj.cast<femx::linalg::ReSolveOptions>();
}

class PythonDeviceTimeIntegrator final
{
public:
  PythonDeviceTimeIntegrator(
      std::unique_ptr<femx::state::DeviceTimeResidual> res,
      femx::linalg::ReSolveOptions                     opts)
    : res_(std::move(res)),
      system_(femx::runtime::makeDeviceLinearSystem(
          femx::runtime::SolverType::ReSolve,
          std::make_unique<femx::linalg::ReSolveLinearSolver>(
              std::move(opts)))),
      integ_(*res_, *system_)
  {
  }

  femx::state::DeviceTimeIntegrator& get() noexcept
  {
    return integ_;
  }

  const femx::state::DeviceTimeIntegrator& get() const noexcept
  {
    return integ_;
  }

  void setInitialState(const HostVector<Real>& init)
  {
    auto& ctx = dynamic_cast<femx::linalg::CudaContext&>(
        system_->context());
    femx::DeviceVector<Real> state;
    ctx.vectors().copy(init, state);
    ctx.sync();
    integ_.setInitialState(state);
  }

  femx::DeviceVector<Real> copyParameters(
      const HostVector<Real>& parameters)
  {
    auto& ctx = dynamic_cast<femx::linalg::CudaContext&>(
        system_->context());
    femx::DeviceVector<Real> out;
    ctx.vectors().copy(parameters, out);
    ctx.sync();
    return out;
  }

private:
  std::unique_ptr<femx::state::DeviceTimeResidual> res_;
  std::unique_ptr<
      femx::linalg::LinearSystem<femx::MemorySpace::Device>>
                                    system_;
  femx::state::DeviceTimeIntegrator integ_;
};

std::unique_ptr<PythonDeviceTimeIntegrator>
makeDeviceIntegrator(NavierStokesModel&     model,
                     FixedDirichletProblem& problem,
                     const RealArray&       initial,
                     const py::object&      options)
{
  auto integrator =
      std::make_unique<PythonDeviceTimeIntegrator>(
          femx::model::ns::makeDeviceTimeResidual(
              model, problem.controlMap()),
          resolveOptions(options));
  const HostVector<Real> host_initial =
      realVector(initial, "initial_state");
  integrator->setInitialState(host_initial);
  return integrator;
}

std::unique_ptr<PythonDeviceTimeIntegrator>
makeDeviceIntegrator(NavierStokesModel&          model,
                     ControlledDirichletProblem& problem,
                     const RealArray&            mean,
                     const RealArray&            modes,
                     const py::object&           options)
{
  const Index num_prm = problem.residual().dims().num_param;
  auto        init    = realVector(mean, "initial_state_mean");
  auto        basis   = realMatrix(modes, "initial_state_modes");
  const auto  opts    = resolveOptions(options);
  if (basis.cols() == 0)
  {
    auto integrator =
        std::make_unique<PythonDeviceTimeIntegrator>(
            femx::model::ns::makeDeviceTimeResidual(
                model, problem.controlMap()),
            opts);
    integrator->setInitialState(init);
    return integrator;
  }
  auto init_map = femx::fem::makeInitialStateMap(
      std::move(init),
      std::move(basis),
      problem.control(),
      0,
      problem.controlParamOffset(),
      num_prm);
  return std::make_unique<PythonDeviceTimeIntegrator>(
      femx::model::ns::makeDeviceTimeResidual(
          model, problem.controlMap(), std::move(init_map)),
      opts);
}

class PythonDeviceTimeReducedFunctional final
  : public PythonTimeReducedFunctional
{
public:
  PythonDeviceTimeReducedFunctional(
      PythonDeviceTimeIntegrator&         owner,
      const femx::inverse::TimeObjective& obj,
      const py::object&                   options)
    : adj_system_(femx::runtime::makeDeviceLinearSystem(
          femx::runtime::SolverType::ReSolve,
          std::make_unique<femx::linalg::ReSolveLinearSolver>(
              resolveOptions(options)))),
      impl_(owner.get(), *adj_system_, obj)
  {
  }

  Index numParams() const noexcept override
  {
    return impl_.numParams();
  }

  Real value(femx::HostVectorView<const Real>   prm,
             femx::inverse::TimeReducedProgress progress = {}) override
  {
    return impl_.value(prm, std::move(progress));
  }

  void grad(femx::HostVectorView<const Real>   prm,
            femx::HostVectorView<Real>         out,
            femx::inverse::TimeReducedProgress progress = {}) override
  {
    impl_.grad(prm, out, std::move(progress));
  }

  Real valueGrad(femx::HostVectorView<const Real>   prm,
                 femx::HostVectorView<Real>         out,
                 femx::inverse::TimeReducedProgress progress = {}) override
  {
    return impl_.valueGrad(prm, out, std::move(progress));
  }

  void resetTiming() noexcept override
  {
    impl_.resetTiming();
  }

  Real assemblySeconds() const noexcept override
  {
    return impl_.assemblySeconds();
  }

  Real solveSeconds() const noexcept override
  {
    return impl_.solveSeconds();
  }

  Index assemblyCalls() const noexcept override
  {
    return impl_.assemblyCalls();
  }

  Index solveCalls() const noexcept override
  {
    return impl_.solveCalls();
  }

private:
  std::unique_ptr<
      femx::linalg::LinearSystem<femx::MemorySpace::Device>>
                                             adj_system_;
  femx::inverse::DeviceTimeReducedFunctional impl_;
};
#endif

} // namespace

void bindNavierStokes(py::module_& module)
{
  py::class_<FluidProperties>(module, "FluidParams")
      .def(py::init<>())
      .def_readwrite("density", &FluidProperties::rho)
      .def_readwrite("dynamic_viscosity", &FluidProperties::mu);

  py::class_<NavierStokesModel>(module, "NavierStokesModel")
      .def(py::init<const std::string&, Index, Real, FluidProperties>(),
           py::arg("mesh_file"),
           py::arg("num_steps"),
           py::arg("dt"),
           py::arg("fluid") = FluidProperties{})
      .def_property_readonly("num_steps", &NavierStokesModel::numSteps)
      .def_property_readonly("num_states", &NavierStokesModel::numStates)
      .def_property_readonly("dt", &NavierStokesModel::dt)
      .def_property_readonly(
          "fluid",
          &NavierStokesModel::fluid,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "mesh",
          &NavierStokesModel::mesh,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "residual",
          [](NavierStokesModel& model) -> TimeResidual&
          {
            return model.residual();
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "velocity_dofs",
          [](const NavierStokesModel& model)
          {
            return indexArray(model.velocityDofs());
          })
      .def("velocity_boundary_dofs",
           &boundaryVelocityDofs,
           py::arg("selector"))
      .def("write_xdmf",
           &writeNavierStokesXdmf,
           py::arg("path"),
           py::arg("trajectory"),
           py::call_guard<py::gil_scoped_release>());

  py::class_<FixedDirichletProblem>(module, "_FixedDirichletProblem")
      .def(py::init<NavierStokesModel&, const py::iterable&>(),
           py::arg("model"),
           py::arg("bcs"),
           py::keep_alive<1, 2>())
      .def_property_readonly(
          "residual",
          &FixedDirichletProblem::residual,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "fixed_dofs",
          [](const FixedDirichletProblem& problem)
          {
            return indexArray(problem.data().dofs);
          })
      .def_property_readonly(
          "fixed_values",
          [](const FixedDirichletProblem& problem)
          {
            return timeDirichletValueArray(problem.data(), problem.numSteps());
          })
      .def_property_readonly(
          "initial_state",
          [](const FixedDirichletProblem& problem)
          {
            py::array_t<Real> out(problem.data().init_state.size());
            auto              vals = out.mutable_unchecked<1>();
            for (Index i = 0; i < problem.data().init_state.size(); ++i)
            {
              vals(i) = problem.data().init_state[i];
            }
            return out;
          });

  py::class_<ControlledDirichletProblem>(
      module, "_ControlledDirichletProblem")
      .def(py::init<NavierStokesModel&,
                    const py::iterable&,
                    const py::object&,
                    Index>(),
           py::arg("model"),
           py::arg("bcs"),
           py::arg("ctr"),
           py::arg("ctr_param_offset") = 0,
           py::keep_alive<1, 2>())
      .def_property_readonly(
          "residual",
          &ControlledDirichletProblem::residual,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "fixed_dofs",
          [](const ControlledDirichletProblem& problem)
          {
            return indexArray(problem.data().dofs);
          })
      .def_property_readonly(
          "fixed_values",
          [](const ControlledDirichletProblem& problem)
          {
            return timeDirichletValueArray(problem.data(), problem.numSteps());
          })
      .def_property_readonly(
          "initial_state",
          [](const ControlledDirichletProblem& problem)
          {
            py::array_t<Real> out(problem.data().init_state.size());
            auto              vals = out.mutable_unchecked<1>();
            for (Index i = 0; i < problem.data().init_state.size(); ++i)
            {
              vals(i) = problem.data().init_state[i];
            }
            return out;
          })
      .def_property_readonly(
          "ctr_state_dofs",
          [](const ControlledDirichletProblem& problem)
          {
            return indexArray(problem.control().stateDofs());
          })
      .def_property_readonly(
          "num_ctr_parameters",
          [](const ControlledDirichletProblem& problem)
          {
            return problem.control().numControlParams();
          })
      .def_property_readonly(
          "num_ctr_levels",
          &ControlledDirichletProblem::numControlLevels)
      .def_property_readonly(
          "ctr_param_offset",
          &ControlledDirichletProblem::controlParamOffset)
      .def_property_readonly(
          "ctr_mesh_node_ids",
          [](const ControlledDirichletProblem& problem)
          {
            return indexArray(problem.controlMeshNodeIds());
          })
      .def("make_initial_state_map",
           &ControlledDirichletProblem::makeInitialStateMap,
           py::arg("mean"),
           py::arg("modes"));

#ifdef FEMX_RESOLVE_USE_CUDA
  py::class_<PythonDeviceTimeIntegrator>(
      module, "_DeviceTimeIntegrator")
      .def(py::init(static_cast<std::unique_ptr<PythonDeviceTimeIntegrator> (*)(
                        NavierStokesModel&,
                        FixedDirichletProblem&,
                        const RealArray&,
                        const py::object&)>(&makeDeviceIntegrator)),
           py::arg("model"),
           py::arg("problem"),
           py::arg("initial_state"),
           py::arg("options") = py::none(),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>())
      .def(py::init(static_cast<std::unique_ptr<PythonDeviceTimeIntegrator> (*)(
                        NavierStokesModel&,
                        ControlledDirichletProblem&,
                        const RealArray&,
                        const RealArray&,
                        const py::object&)>(&makeDeviceIntegrator)),
           py::arg("model"),
           py::arg("problem"),
           py::arg("initial_state_mean"),
           py::arg("initial_state_modes"),
           py::arg("options") = py::none(),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>())
      .def_property_readonly(
          "num_steps",
          [](const PythonDeviceTimeIntegrator& owner)
          { return owner.get().numSteps(); })
      .def_property_readonly(
          "num_states",
          [](const PythonDeviceTimeIntegrator& owner)
          { return owner.get().numStates(); })
      .def_property_readonly(
          "num_param",
          [](const PythonDeviceTimeIntegrator& owner)
          { return owner.get().numParams(); })
      .def(
          "solve",
          [](PythonDeviceTimeIntegrator& owner,
             const RealArray&            parameters,
             const py::object&           progress)
          {
            auto& integrator = owner.get();
            if (!progress.is_none() && !PyCallable_Check(progress.ptr()))
            {
              throw py::type_error("progress must be callable");
            }
            const HostVector<Real> values =
                realVector(parameters, "parameters");
            auto           device_values = owner.copyParameters(values);
            TimeTrajectory trajectory;
            if (progress.is_none())
            {
              py::gil_scoped_release release;
              integrator.solve(device_values.view(), trajectory);
            }
            else
            {
              femx::state::DeviceTimeIntegrator::Observer observer =
                  [&progress](
                      const femx::state::TimeStepStateContext& step)
              {
                py::gil_scoped_acquire acquire;
                if (PyErr_CheckSignals() != 0)
                {
                  throw py::error_already_set();
                }
                if (step.level == 0)
                {
                  return false;
                }
                py::dict event;
                event["type"]                 = "solve";
                event["phase"]                = "forward";
                event["step"]                 = step.level;
                event["total"]                = step.total_steps;
                event["assembly_seconds"]     = step.assm_sec;
                event["linear_solve_seconds"] = step.lin_solve_sec;
                progress(std::move(event));
                return false;
              };
              py::gil_scoped_release release;
              integrator.solve(device_values.view(),
                               trajectory,
                               std::move(observer));
            }
            return trajectory;
          },
          py::arg("param"),
          py::arg("progress") = py::none())
      .def("reset_timing",
           [](PythonDeviceTimeIntegrator& owner)
           { owner.get().resetStats(); })
      .def_property_readonly(
          "assembly_seconds",
          [](const PythonDeviceTimeIntegrator& owner)
          { return owner.get().lastStats().assm_sec; })
      .def_property_readonly(
          "solve_seconds",
          [](const PythonDeviceTimeIntegrator& owner)
          { return owner.get().lastStats().lin_solve_sec; })
      .def_property_readonly(
          "assembly_calls",
          [](const PythonDeviceTimeIntegrator& owner)
          { return owner.get().lastStats().assm_calls; })
      .def_property_readonly(
          "solve_calls",
          [](const PythonDeviceTimeIntegrator& owner)
          { return owner.get().lastStats().lin_solve_calls; });

  py::class_<PythonDeviceTimeReducedFunctional,
             PythonTimeReducedFunctional>(
      module, "_DeviceTimeReducedFunctional")
      .def(py::init<PythonDeviceTimeIntegrator&,
                    const femx::inverse::TimeObjective&,
                    const py::object&>(),
           py::arg("integrator"),
           py::arg("objective"),
           py::arg("options") = py::none(),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>());
#endif

  py::class_<VelocityPointSampler,
             femx::inverse::TimeObservationOperator>(
      module, "_VelocityPointSampler")
      .def(py::init<const NavierStokesModel&,
                    const RealArray&,
                    const IndexArray&,
                    Index>(),
           py::arg("model"),
           py::arg("points"),
           py::arg("components"),
           py::arg("num_param") = 0,
           py::keep_alive<1, 2>())
      .def("sample", &VelocityPointSampler::sample, py::arg("trajectory"))
      .def("apply_transpose",
           &VelocityPointSampler::applyTranspose,
           py::arg("solve_level_directions"));
}
