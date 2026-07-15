#pragma once

#include <stdexcept>
#include <utility>

#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

class InitialStateParameterMap
{
public:
  using Index = femx::Index;
  using Real  = femx::Real;

  InitialStateParameterMap(femx::Vector<Real>          mean,
                           femx::DenseMatrix           modes,
                           femx::fem::DirichletControl control,
                           Index                       init_offset,
                           Index                       ctr_offset,
                           Index                       num_param)
    : mean_(std::move(mean)),
      modes_(std::move(modes)),
      control_(std::move(control)),
      init_offset_(init_offset),
      ctr_offset_(ctr_offset),
      num_param_(num_param)
  {
    if (mean_.empty() || modes_.rows() != mean_.size())
    {
      throw std::runtime_error(
          "initial-state mean and modes must match the state size");
    }
    if (init_offset_ < 0 || ctr_offset_ < 0 || num_param_ < 0
        || init_offset_ + modes_.cols() > num_param_
        || ctr_offset_ + control_.numControlParams() > num_param_)
    {
      throw std::runtime_error(
          "initial-state parameter blocks do not fit the parameter vector");
    }
    for (Index row : control_.stateDofs())
    {
      if (row < 0 || row >= mean_.size())
      {
        throw std::runtime_error(
            "initial-state control row is out of range");
      }
      for (Index col = 0; col < modes_.cols(); ++col)
      {
        if (modes_(row, col) != 0.0)
        {
          throw std::runtime_error(
              "initial-state modes must vanish on control rows");
        }
      }
    }
  }

  Index numStates() const
  {
    return mean_.size();
  }

  Index numParams() const
  {
    return num_param_;
  }

  Index numModes() const
  {
    return modes_.cols();
  }

  void state(const femx::Vector<Real>& param,
             femx::Vector<Real>&       out) const
  {
    checkParam(param);
    out = mean_;
    for (Index row = 0; row < modes_.rows(); ++row)
    {
      for (Index col = 0; col < modes_.cols(); ++col)
      {
        out[row] += modes_(row, col) * param[init_offset_ + col];
      }
    }

    femx::Vector<Real> ctr_param(control_.numControlParams());
    for (Index i = 0; i < ctr_param.size(); ++i)
    {
      ctr_param[i] = param[ctr_offset_ + i];
    }
    femx::Vector<Real> values;
    control_.apply(ctr_param, values);
    for (Index i = 0; i < control_.numStateDofs(); ++i)
    {
      out[control_.stateDof(i)] = values[i];
    }
  }

  void applyTranspose(const femx::Vector<Real>& state_grad,
                      femx::Vector<Real>&       out) const
  {
    if (state_grad.size() != numStates())
    {
      throw std::runtime_error(
          "initial-state gradient size does not match the state size");
    }
    out.resize(num_param_);
    out.setZero();
    for (Index col = 0; col < modes_.cols(); ++col)
    {
      for (Index row = 0; row < modes_.rows(); ++row)
      {
        out[init_offset_ + col] += modes_(row, col) * state_grad[row];
      }
    }

    femx::Vector<Real> control_state_grad(control_.numStateDofs());
    for (Index i = 0; i < control_.numStateDofs(); ++i)
    {
      control_state_grad[i] = state_grad[control_.stateDof(i)];
    }
    femx::Vector<Real> control_grad;
    control_.applyTranspose(control_state_grad, control_grad);
    for (Index i = 0; i < control_grad.size(); ++i)
    {
      out[ctr_offset_ + i] += control_grad[i];
    }
  }

private:
  void checkParam(const femx::Vector<Real>& param) const
  {
    if (param.size() != num_param_)
    {
      throw std::runtime_error(
          "initial-state parameter size mismatch");
    }
  }

private:
  femx::Vector<Real>          mean_;
  femx::DenseMatrix           modes_;
  femx::fem::DirichletControl control_;
  Index                       init_offset_{0};
  Index                       ctr_offset_{0};
  Index                       num_param_{0};
};
