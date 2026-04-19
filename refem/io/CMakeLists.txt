add_library(refem_io)
add_library(refem::io ALIAS refem_io)

target_sources(refem_io
  PRIVATE
    DataOut.cpp
    Hdf5Writer.cpp
    XdmfWriter.cpp
  PUBLIC
    DataOut.hpp
    Hdf5Writer.hpp
    XdmfWriter.hpp
)

target_link_libraries(refem_io
  PUBLIC
    refem_config
    refem_common
    refem_linalg
    refem_mesh
)

if(HDF5_FOUND)
  target_include_directories(refem_io PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(refem_io PRIVATE ${HDF5_LIBRARIES})

  if(HDF5_IS_PARALLEL AND MPI_CXX_FOUND)
    target_link_libraries(refem_io PRIVATE MPI::MPI_CXX)
  endif()

  if(REFEM_MPI_INCLUDE_DIR)
    target_include_directories(refem_io PRIVATE ${REFEM_MPI_INCLUDE_DIR})
  endif()
endif()
