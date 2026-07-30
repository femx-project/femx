#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <pybind11/numpy.h>

namespace femx::python::bindings
{

using RealArray =
    pybind11::array_t<Real,
                      pybind11::array::c_style
                          | pybind11::array::forcecast>;
using IndexArray =
    pybind11::array_t<Index,
                      pybind11::array::c_style
                          | pybind11::array::forcecast>;

enum class FiniteCheck
{
  Skip,
  Require
};

inline void requireFinite(HostVectorView<const Real> vals,
                          const char*                name)
{
  for (Real value : vals)
  {
    if (!std::isfinite(value))
    {
      throw std::runtime_error(std::string(name) + " must be finite");
    }
  }
}

inline void requireFinite(const HostVector<Real>& vals, const char* name)
{
  requireFinite(vals.view(), name);
}

inline void requireFinite(const DenseMatrix& vals, const char* name)
{
  for (Index row = 0; row < vals.rows(); ++row)
  {
    for (Index col = 0; col < vals.cols(); ++col)
    {
      if (!std::isfinite(vals(row, col)))
      {
        throw std::runtime_error(std::string(name) + " must be finite");
      }
    }
  }
}

inline HostVector<Real> vectorFromArray(const RealArray& vals,
                                        const char*      name,
                                        FiniteCheck      finite_check)
{
  if (vals.ndim() != 1)
  {
    throw std::runtime_error(std::string(name) + " must be one-dimensional");
  }

  HostVector<Real> out(static_cast<Index>(vals.shape(0)));
  const auto       data = vals.unchecked<1>();
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = data(i);
  }
  if (finite_check == FiniteCheck::Require)
  {
    requireFinite(out, name);
  }
  return out;
}

inline HostVector<Index> indexVectorFromArray(const IndexArray& vals,
                                              const char*       name)
{
  if (vals.ndim() != 1)
  {
    throw std::runtime_error(std::string(name) + " must be one-dimensional");
  }

  HostVector<Index> out(static_cast<Index>(vals.shape(0)));
  const auto        data = vals.unchecked<1>();
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = data(i);
  }
  return out;
}

inline DenseMatrix denseMatrixFromArray(const RealArray& vals,
                                        const char*      name,
                                        FiniteCheck      finite_check)
{
  if (vals.ndim() != 2)
  {
    throw std::runtime_error(
        std::string(name) + " must be two-dimensional");
  }

  DenseMatrix out(static_cast<Index>(vals.shape(0)),
                  static_cast<Index>(vals.shape(1)));
  const auto  data = vals.unchecked<2>();
  for (Index row = 0; row < out.rows(); ++row)
  {
    for (Index col = 0; col < out.cols(); ++col)
    {
      out(row, col) = data(row, col);
    }
  }
  if (finite_check == FiniteCheck::Require)
  {
    requireFinite(out, name);
  }
  return out;
}

inline HostVector<Real> flattenedVectorFromArray(
    const RealArray& vals,
    Index            expected_size,
    const char*      name,
    FiniteCheck      finite_check)
{
  if (vals.size() != expected_size)
  {
    throw std::runtime_error(
        std::string(name) + " has an inconsistent size");
  }

  HostVector<Real> out(expected_size);
  const Real*      data = vals.data();
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = data[i];
  }
  if (finite_check == FiniteCheck::Require)
  {
    requireFinite(out, name);
  }
  return out;
}

template <class T>
pybind11::array_t<T> vectorArray(HostVectorView<const T> vals)
{
  pybind11::array_t<T> out(vals.size());
  auto                 data = out.template mutable_unchecked<1>();
  for (Index i = 0; i < vals.size(); ++i)
  {
    data(i) = vals[i];
  }
  return out;
}

template <class T>
pybind11::array_t<T> vectorArray(const HostVector<T>& vals)
{
  return vectorArray(vals.view());
}

inline pybind11::array_t<Real> denseMatrixArray(const DenseMatrix& vals)
{
  pybind11::array_t<Real> out({vals.rows(), vals.cols()});
  auto                    data = out.mutable_unchecked<2>();
  for (Index row = 0; row < vals.rows(); ++row)
  {
    for (Index col = 0; col < vals.cols(); ++col)
    {
      data(row, col) = vals(row, col);
    }
  }
  return out;
}

} // namespace femx::python::bindings
