#pragma once

#include <utility>

#include <femx/common/Context.hpp>
#include <femx/common/LinearInterpolation.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>

namespace femx
{
namespace fem
{

/**
 * @brief Fixed and controlled Dirichlet data in one memory space.
 *
 * The spatial control map is stored once as a compact CSR matrix. Host and
 * Device operations use the same linalg apply/applyT and gather/scatter API.
 */
template <MemorySpace Space>
class ControlMap
{
public:
  ControlMap() = default;

  ControlMap(const ControlMap&)                = default;
  ControlMap(ControlMap&&) noexcept            = default;
  ControlMap& operator=(const ControlMap&)     = default;
  ControlMap& operator=(ControlMap&&) noexcept = default;

  Index numSteps() const noexcept
  {
    return num_steps_;
  }

  Index numStates() const noexcept
  {
    return num_states_;
  }

  Index numParams() const noexcept
  {
    return num_prm_;
  }

  Index numControlParams() const noexcept
  {
    return control_.cols();
  }

  Index numControlledDofs() const noexcept
  {
    return control_.rows();
  }

  Index numBcs() const noexcept
  {
    return dofs_.size();
  }

  const Vector<Space, Index>& dofs() const noexcept
  {
    return dofs_;
  }

  const CsrMatrix<Space>& controlMatrix() const noexcept
  {
    return control_;
  }

private:
  friend ControlMap<MemorySpace::Host> makeControlMap(
      Index,
      Index,
      const DirichletControl&,
      Array<Index>,
      HostVector,
      Array<LinearInterpolation>,
      Index,
      Index);

  friend void copy(const ControlMap<MemorySpace::Host>&,
                   ControlMap<MemorySpace::Device>&,
                   CudaContext&);

  friend void controlVals(const ControlMap<MemorySpace::Host>&,
                          Index,
                          HostConstVectorView,
                          HostVectorView);
  friend void controlVals(const ControlMap<MemorySpace::Device>&,
                          Index,
                          DeviceConstVectorView,
                          DeviceVectorView,
                          CudaContext&);
  friend void controlJac(const ControlMap<MemorySpace::Host>&,
                         Index,
                         HostConstVectorView,
                         HostVectorView);
  friend void controlJac(const ControlMap<MemorySpace::Device>&,
                         Index,
                         DeviceConstVectorView,
                         DeviceVectorView,
                         CudaContext&);
  friend void addControlJacT(const ControlMap<MemorySpace::Host>&,
                             Index,
                             HostConstVectorView,
                             HostVectorView);
  friend void addControlJacT(const ControlMap<MemorySpace::Device>&,
                             Index,
                             DeviceConstVectorView,
                             DeviceVectorView,
                             CudaContext&);

  Index num_steps_{0};
  Index num_states_{0};
  Index num_prm_{0};
  Index num_fixed_{0};
  Index ctr_off_{0};

  CsrMatrix<Space>            control_;
  Vector<Space, Index>        dofs_;
  Vector<Space, Real>         fixed_vals_;
  // Time interpolation is orchestration metadata consumed on Host even when
  // the matrix and vectors live on Device.
  HostIndexVector             lower_;
  HostIndexVector             upper_;
  HostVector                  upper_wts_;
  mutable Vector<Space, Real> compact_;
};

using HostControlMap   = ControlMap<MemorySpace::Host>;
using DeviceControlMap = ControlMap<MemorySpace::Device>;

HostControlMap makeControlMap(
    Index                      num_steps,
    Index                      num_states,
    const DirichletControl&    ctr,
    Array<Index>               fixed_dofs,
    HostVector                 fixed_vals,
    Array<LinearInterpolation> time,
    Index                      ctr_off,
    Index                      num_prm = -1);

void copy(const HostControlMap& src,
          DeviceControlMap&     dst,
          CudaContext&          ctx);

void controlVals(const HostControlMap& map,
                 Index                 step,
                 HostConstVectorView   prm,
                 HostVectorView        out);

void controlVals(const DeviceControlMap& map,
                 Index                   step,
                 DeviceConstVectorView   prm,
                 DeviceVectorView        out,
                 CudaContext&            ctx);

void controlJac(const HostControlMap& map,
                Index                 step,
                HostConstVectorView   dir,
                HostVectorView        out);

void controlJac(const DeviceControlMap& map,
                Index                   step,
                DeviceConstVectorView   dir,
                DeviceVectorView        out,
                CudaContext&            ctx);

void addControlJacT(const HostControlMap& map,
                    Index                 step,
                    HostConstVectorView   adj,
                    HostVectorView        grad);

void addControlJacT(const DeviceControlMap& map,
                    Index                   step,
                    DeviceConstVectorView   adj,
                    DeviceVectorView        grad,
                    CudaContext&            ctx);

/**
 * @brief Affine initial-state parameterization in one memory space.
 *
 * Modes are row-major dense data and the controlled part is a compact CSR
 * matrix. Device evaluation therefore uses cuBLAS plus cuSPARSE.
 */
template <MemorySpace Space>
class InitialStateMap
{
public:
  InitialStateMap() = default;

  InitialStateMap(const InitialStateMap&)                = default;
  InitialStateMap(InitialStateMap&&) noexcept            = default;
  InitialStateMap& operator=(const InitialStateMap&)     = default;
  InitialStateMap& operator=(InitialStateMap&&) noexcept = default;

  Index numStates() const noexcept
  {
    return num_states_;
  }

  Index numParams() const noexcept
  {
    return num_prm_;
  }

  Index numModes() const noexcept
  {
    return num_modes_;
  }

private:
  friend InitialStateMap<MemorySpace::Host> makeInitialStateMap(
      HostVector,
      DenseMatrix,
      const DirichletControl&,
      Index,
      Index,
      Index);

  friend void copy(const InitialStateMap<MemorySpace::Host>&,
                   InitialStateMap<MemorySpace::Device>&,
                   CudaContext&);

  friend void initialState(const InitialStateMap<MemorySpace::Host>&,
                           HostConstVectorView,
                           HostVectorView);
  friend void initialState(const InitialStateMap<MemorySpace::Device>&,
                           DeviceConstVectorView,
                           DeviceVectorView,
                           CudaContext&);
  friend void addInitialJacT(const InitialStateMap<MemorySpace::Host>&,
                             HostConstVectorView,
                             HostVectorView);
  friend void addInitialJacT(const InitialStateMap<MemorySpace::Device>&,
                             DeviceConstVectorView,
                             DeviceVectorView,
                             CudaContext&);

  Index num_states_{0};
  Index num_prm_{0};
  Index num_modes_{0};
  Index init_off_{0};
  Index ctr_off_{0};

  Vector<Space, Real>         mean_;
  Vector<Space, Real>         modes_;
  CsrMatrix<Space>            control_;
  Vector<Space, Index>        ctr_dofs_;
  mutable Vector<Space, Real> compact_;
};

using HostInitialStateMap   = InitialStateMap<MemorySpace::Host>;
using DeviceInitialStateMap = InitialStateMap<MemorySpace::Device>;

HostInitialStateMap makeInitialStateMap(HostVector              mean,
                                        DenseMatrix             modes,
                                        const DirichletControl& ctr,
                                        Index                   init_off,
                                        Index                   ctr_off,
                                        Index                   num_prm);

void copy(const HostInitialStateMap& src,
          DeviceInitialStateMap&     dst,
          CudaContext&               ctx);

void initialState(const HostInitialStateMap& map,
                  HostConstVectorView        prm,
                  HostVectorView             out);

void initialState(const DeviceInitialStateMap& map,
                  DeviceConstVectorView        prm,
                  DeviceVectorView             out,
                  CudaContext&                 ctx);

void addInitialJacT(const HostInitialStateMap& map,
                    HostConstVectorView        adj,
                    HostVectorView             grad);

void addInitialJacT(const DeviceInitialStateMap& map,
                    DeviceConstVectorView        adj,
                    DeviceVectorView             grad,
                    CudaContext&                 ctx);

} // namespace fem
} // namespace femx
