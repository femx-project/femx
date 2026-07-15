#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include "Bindings.hpp"
#include "InitialStateParameterMap.hpp"
#include "PETScInit.hpp"
#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/common/LinearInterpolation.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/TimeDirichletData.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/Config.hpp>
#include <femx/model/ns/Helper.hpp>
#include <femx/model/ns/NavierStokesModel.hpp>
#ifdef FEMX_HAS_PETSC
#include <femx/runtime/PETScRuntime.hpp>
#endif
#include <femx/state/TimeResidual.hpp>
#include <femx/state/TimeTrajectory.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{

using femx::DenseMatrix;
using femx::Index;
using femx::Real;
using femx::Vector;
using femx::assembly::TimeDirichletControlResidual;
using femx::model::ns::FluidParams;
using femx::model::ns::NavierStokesModel;
using femx::state::TimeResidual;
using femx::state::TimeTrajectory;

using RealArray  = py::array_t<Real,
                               py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<Index,
                               py::array::c_style | py::array::forcecast>;

py::array_t<Index> indexArray(const Vector<Index>& values)
{
  py::array_t<Index> out(values.size());
  auto               data = out.mutable_unchecked<1>();
  for (Index i = 0; i < values.size(); ++i)
  {
    data(i) = values[i];
  }
  return out;
}

Vector<Real> realVector(const RealArray& values, const char* name)
{
  if (values.ndim() != 1)
  {
    throw std::runtime_error(std::string(name) + " must be one-dimensional");
  }
  Vector<Real> out(values.shape(0));
  const auto   data = values.unchecked<1>();
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

DenseMatrix realMatrix(const RealArray& values, const char* name)
{
  if (values.ndim() != 2)
  {
    throw std::runtime_error(std::string(name) + " must be two-dimensional");
  }
  DenseMatrix out(values.shape(0), values.shape(1));
  const auto  data = values.unchecked<2>();
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

  const Index  num_nodes = model.mesh().numNodes();
  Vector<Real> ux(num_nodes);
  Vector<Real> uy(num_nodes);
  Vector<Real> uz(num_nodes);
  Vector<Real> pressure(num_nodes);

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
  auto              values = out.mutable_unchecked<1>();
  for (Index component = 0; component < dim; ++component)
  {
    values(component) = point[component];
  }
  return out;
}

Vector<Real> pythonBoundaryValue(const py::object&            value,
                                 const femx::fem::Mesh::Node& point,
                                 Index                        dim,
                                 Real                         time,
                                 Index                        components,
                                 const std::string&           field)
{
  const py::object raw    = PyCallable_Check(value.ptr())
                                ? value(pointArray(point, dim), time)
                                : value;
  const RealArray  values = RealArray::ensure(raw);
  if (!values)
  {
    throw std::runtime_error(field + " boundary value must be real-valued");
  }

  Vector<Real> out(components);
  if (components == 1 && values.ndim() == 0)
  {
    out[0] = *values.data();
  }
  else if (values.ndim() == 1 && values.shape(0) == components)
  {
    const auto data = values.unchecked<1>();
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
      const Vector<Real> components = pythonBoundaryValue(
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

Vector<Real> pythonNormal(const NavierStokesModel& model,
                          const py::object&        specification)
{
  const RealArray values =
      RealArray::ensure(specification.attr("normal"));
  const Index components = model.space().field(fieldId("velocity")).numComponents();
  if (!values || values.ndim() != 1 || values.shape(0) != components)
  {
    throw std::runtime_error(
        "normal must contain one value per velocity component");
  }

  Vector<Real> normal(components);
  const auto   data = values.unchecked<1>();
  for (Index component = 0; component < components; ++component)
  {
    normal[component] = data(component);
  }
  return normal;
}

femx::fem::DirichletControl makePythonNormalControl(
    const NavierStokesModel& model,
    const py::object&        specification,
    const Vector<Index>&     fixed_dofs)
{
  const py::object            selector = specification.attr("boundary");
  const Vector<Real>          normal   = pythonNormal(model, specification);
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
    const Vector<Index>&     fixed_dofs)
{
  const py::object selector = specification.attr("boundary");
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
    const Vector<Index>&     fixed_dofs)
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

Vector<femx::LinearInterpolation> makePythonControlTimeStencils(
    const NavierStokesModel& model,
    const py::object&        specification)
{
  const py::object raw_times = specification.attr("times");
  if (raw_times.is_none())
  {
    return {};
  }

  const RealArray values = RealArray::ensure(raw_times);
  if (!values || values.ndim() != 1 || values.shape(0) == 0)
  {
    throw std::runtime_error(
        "control times must be a nonempty one-dimensional array");
  }

  Vector<Real> times(values.shape(0));
  const auto   data = values.unchecked<1>();
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

  Vector<femx::LinearInterpolation> stencils(model.numSteps());
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

Vector<femx::Point3> pythonObservationPoints(
    const NavierStokesModel& model,
    const RealArray&         values)
{
  const Index dim = model.mesh().dim();
  if (values.ndim() != 2 || values.shape(0) == 0
      || (values.shape(1) != dim && values.shape(1) != 3))
  {
    throw std::runtime_error(
        "observation points must have shape (num_points, mesh_dimension) or "
        "(num_points, 3)");
  }

  Vector<femx::Point3> points(values.shape(0));
  const auto           data = values.unchecked<2>();
  for (Index point = 0; point < points.size(); ++point)
  {
    points[point] = {0.0, 0.0, 0.0};
    for (Index axis = 0; axis < values.shape(1); ++axis)
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

Vector<Index> pythonObservationComponents(
    const NavierStokesModel& model,
    const IndexArray&        values)
{
  if (values.ndim() != 1 || values.shape(0) == 0)
  {
    throw std::runtime_error(
        "observation components must be a nonempty one-dimensional array");
  }

  const Index     num_components = model.space().field(0).numComponents();
  Vector<Index>   components(values.shape(0));
  std::set<Index> seen;
  const auto      data = values.unchecked<1>();
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
  auto              values = out.mutable_unchecked<2>();
  for (Index step = 0; step < steps; ++step)
  {
    for (Index column = 0; column < data.dofs.size(); ++column)
    {
      values(step, column) = data.values[step * data.dofs.size() + column];
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
      residual_(model_.residual(),
                femx::fem::DirichletControl{},
                data_.dofs,
                0,
                0,
                data_.values)
  {
  }

  TimeResidual& residual()
  {
    return residual_;
  }

  const femx::fem::TimeDirichletData& data() const
  {
    return data_;
  }

  Index numSteps() const
  {
    return model_.numSteps();
  }

private:
  NavierStokesModel&           model_;
  femx::fem::TimeDirichletData data_;
  TimeDirichletControlResidual residual_;
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
      residual_(model_.residual(),
                ctr_,
                data_.dofs,
                ctr_param_offset,
                -1,
                data_.values,
                time_stencils_),
      ctr_param_offset_(ctr_param_offset)
  {
  }

  TimeResidual& residual()
  {
    return residual_;
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
    return (residual_.numParams() - ctr_param_offset_)
           / ctr_.numControlParams();
  }

  Index controlParamOffset() const
  {
    return ctr_param_offset_;
  }

  Vector<Index> controlMeshNodeIds() const
  {
    const auto    u_dof = model_.space().field(0);
    const Index   comps = u_dof.numComponents();
    Vector<Index> nodes(ctr_.numControlParams(), -1);
    for (const auto& entry : ctr_.mapEntries())
    {
      const Index dof  = ctr_.stateDof(entry.state_row);
      const Index node = dof / comps;
      const Index comp = dof % comps;
      if (node < 0 || node >= model_.mesh().numNodes()
          || u_dof.globalDof(node, comp) != dof)
      {
        throw std::runtime_error(
            "Normal velocity control contains a non-velocity dof");
      }
      Index& stored = nodes[entry.ctr_col];
      if (stored >= 0 && stored != node)
      {
        throw std::runtime_error(
            "Normal velocity control column spans multiple nodes");
      }
      stored = node;
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

  std::unique_ptr<InitialStateParameterMap> makeInitialStateMap(
      const RealArray& mean,
      const RealArray& modes) const
  {
    return std::make_unique<InitialStateParameterMap>(
        realVector(mean, "initial_state_mean"),
        realMatrix(modes, "initial_state_modes"),
        ctr_,
        0,
        ctr_param_offset_,
        residual_.numParams());
  }

private:
  NavierStokesModel&                model_;
  femx::fem::TimeDirichletData      data_;
  femx::fem::DirichletControl       ctr_;
  Vector<femx::LinearInterpolation> time_stencils_;
  TimeDirichletControlResidual      residual_;
  Index                             ctr_param_offset_{0};
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

  void observe(Index               level,
               const Vector<Real>& state,
               const Vector<Real>& prm,
               Vector<Real>&       out) const override
  {
    interpolator_.observe(level, state, prm, out);
  }

  void applyStateJac(Index               level,
                     const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    interpolator_.applyStateJac(level, state, prm, dir, out);
  }

  void applyStateJacT(Index               level,
                      const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
  {
    interpolator_.applyStateJacT(level, state, prm, dir, out);
  }

  void applyParamJac(Index               level,
                     const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    interpolator_.applyParamJac(level, state, prm, dir, out);
  }

  void applyParamJacT(Index               level,
                      const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
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
    auto         data = out.mutable_unchecked<3>();
    Vector<Real> parameters(numParams());
    for (Index level = 0; level < trajectory.numTimeLevels(); ++level)
    {
      Vector<Real> values;
      observe(level, trajectory[level], parameters, values);
      for (Index point = 0; point < num_points; ++point)
      {
        for (Index component = 0; component < num_components; ++component)
        {
          data(level, point, component) =
              values[point * num_components + component];
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
    Vector<Real>      state(numStates());
    Vector<Real>      parameters(numParams());
    for (Index level = 0; level <= numSteps(); ++level)
    {
      Vector<Real> direction(numObservations());
      for (Index i = 0; i < numObservations(); ++i)
      {
        direction[i] = input(level, i);
        if (!std::isfinite(direction[i]))
        {
          throw std::runtime_error(
              "observation directions must be finite");
        }
      }

      Vector<Real> gradient;
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

} // namespace

void bindNavierStokes(py::module_& module)
{
  py::class_<FluidParams>(module, "FluidParams")
      .def(py::init<>())
      .def_readwrite("density", &FluidParams::rho)
      .def_readwrite("dynamic_viscosity", &FluidParams::mu);

  py::class_<NavierStokesModel>(module, "NavierStokesModel")
      .def(py::init<const std::string&, Index, Real, FluidParams>(),
           py::arg("mesh_file"),
           py::arg("num_steps"),
           py::arg("dt"),
           py::arg("fluid") = FluidParams{})
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
          "_matrix_pattern",
          &NavierStokesModel::matrixPattern,
          py::return_value_policy::reference_internal)
      .def(
          "_use_full_element_range",
          [](NavierStokesModel& model)
          {
            model.residual().setElemRange(0, model.mesh().numElems());
          })
#ifdef FEMX_HAS_PETSC
      .def(
          "_use_petsc_world_element_range",
          [](NavierStokesModel& model)
          {
            femx::python::initializePETSc();
            femx::runtime::setElemRange(
                model.residual(),
                model.mesh().numElems(),
                PETSC_COMM_WORLD);
          })
#endif
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
            py::array_t<Real> out(problem.data().initial_state.size());
            auto              values = out.mutable_unchecked<1>();
            for (Index i = 0; i < problem.data().initial_state.size(); ++i)
            {
              values(i) = problem.data().initial_state[i];
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
            py::array_t<Real> out(problem.data().initial_state.size());
            auto              values = out.mutable_unchecked<1>();
            for (Index i = 0; i < problem.data().initial_state.size(); ++i)
            {
              values(i) = problem.data().initial_state[i];
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
