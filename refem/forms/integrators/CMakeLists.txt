add_library(refem_forms_integrators)
add_library(refem::forms_integrators ALIAS refem_forms_integrators)

target_sources(refem_forms_integrators
  PRIVATE
    DiffusionIntegrator.cpp
  PUBLIC
    DiffusionIntegrator.hpp
    DomainIntegrator.hpp
)

target_link_libraries(refem_forms_integrators
  PUBLIC
    refem_common
    refem_config
    refem_fe
    refem_linalg
)
