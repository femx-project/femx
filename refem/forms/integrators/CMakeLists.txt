add_library(refem_forms_integrators)
add_library(refem::forms_integrators ALIAS refem_forms_integrators)

target_sources(
  refem_forms_integrators
  PRIVATE AdvectionIntegrator.cpp DiffusionIntegrator.cpp MassIntegrator.cpp
          SourceIntegrator.cpp
  PUBLIC AdvectionIntegrator.hpp DiffusionIntegrator.hpp
         DomainBilinearIntegrator.hpp DomainLinearIntegrator.hpp
         MassIntegrator.hpp SourceIntegrator.hpp)

target_link_libraries(refem_forms_integrators PUBLIC refem_common refem_config
                                                     refem_fe refem_linalg)
