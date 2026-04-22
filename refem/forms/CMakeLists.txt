add_library(refem_forms)
add_library(refem::forms ALIAS refem_forms)

target_sources(
  refem_forms
  PRIVATE BilinearForm.cpp LinearForm.cpp
  PUBLIC BilinearForm.hpp LinearForm.hpp)

add_subdirectory(integrators)

target_link_libraries(refem_forms PUBLIC refem_common refem_config refem_fe
                                         refem_forms_integrators refem_linalg)
